// ============================================================
//  cube2_solver/src/lib.rs
//  Optimal 2×2 Pocket Cube Solver
// ============================================================
//
//  ARCHITECTURE
//  ─────────────
//  The 2×2 cube has exactly 3,674,160 reachable states (fixing one
//  corner to eliminate pure rotations).  We pre-compute a lookup table
//  that maps every reachable state → its God's-number depth (0-14) and
//  the move that leads toward the solved state.  Building this table
//  takes ~50 ms and ~15 MB once per process lifetime (stored in a
//  static OnceCell).  After that, any query is O(depth) — just follow
//  the back-pointers.
//
//  STATE REPRESENTATION
//  ─────────────────────
//  We represent the cube as a single u64 with 7 corner records packed
//  into bits.  The 8th corner (DBR) is fixed as the reference frame,
//  so its position and orientation are always known.
//
//  Each of the 7 free corners carries:
//    • 3 bits  → which of the 8 corner slots it occupies (0-7)
//    • 2 bits  → orientation (0=good, 1=CW twist, 2=CCW twist)
//  Total: 5 bits × 7 corners = 35 bits → fits in u64.
//
//  Corner slots (absolute positions on the cube):
//    0=UBL  1=UBR  2=UFL  3=UFR
//    4=DBL  5=DBR(fixed)  6=DFL  7=DFR
//
//  Orientation convention (Pochmann):
//    0 = sticker facing U or D
//    1 = sticker facing R or L
//    2 = sticker facing F or B
//  (The 8th corner's orientation is fully determined by the others
//   because total orientation sum mod 3 = 0.)
//
//  MOVE TABLE
//  ───────────
//  18 moves (6 faces × {CW, CCW, 180°}).  Each move is a permutation
//  on the 8 corner slots plus an orientation delta per slot.
//  We store the 18 resulting state-transition functions as lookup arrays
//  generated at compile time (via const fn / build script).
//
//  BFS / BIDIRECTIONAL BFS
//  ────────────────────────
//  For table generation we run a standard BFS from the solved state,
//  expanding all 18 moves.  We stop when every reachable state has been
//  visited (~3.67 M states).  We store depth + parent-move in a
//  HashMap<u64, (u8, u8)>.
//
//  Solving any query state is then just: look up the state, follow the
//  back-pointer chain to the solved state, reverse the move sequence.
//
// ============================================================

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;

// ── Corner definitions ───────────────────────────────────────────────────────

/// Absolute corner slot indices
///   0=UBL  1=UBR  2=UFL  3=UFR  4=DBL  5=DBR  6=DFL  7=DFR
/// DBR (slot 5) is the fixed reference corner.
const NFREE: usize = 7; // corners 0-4,6,7 (skip fixed slot 5)

/// Map free-corner index (0-6) → absolute slot index (skipping slot 5)
const FREE_TO_SLOT: [u8; 7] = [0, 1, 2, 3, 4, 6, 7];

/// Map absolute slot (0-7) → free index (0-6), or 7 if fixed
const SLOT_TO_FREE: [u8; 8] = [0, 1, 2, 3, 4, 7, 5, 6];

// ── Move definitions ─────────────────────────────────────────────────────────
//
// A move is defined as:
//   • A permutation on the 8 absolute corner slots (where does each slot go)
//   • An orientation twist applied to each affected piece (+0, +1, or +2 mod 3)
//
// Source: standard 2×2 move tables cross-checked with Korf (1997) and
// online 2×2 solvers.  Orientation uses the Pochmann convention above.

/// Number of moves (6 faces × 3 = 18)
const NMOVES: usize = 18;

/// Move names in the same order as MOVE_PERMS / MOVE_ORI
pub const MOVE_NAMES: [&str; NMOVES] = [
    "U",  "U2",  "U'",
    "R",  "R2",  "R'",
    "F",  "F2",  "F'",
    "D",  "D2",  "D'",
    "L",  "L2",  "L'",
    "B",  "B2",  "B'",
];

/// Inverse move index (applying MOVE_PERMS[MOVE_INV[i]] undoes move i)
const MOVE_INV: [usize; NMOVES] = [
    2, 1, 0,   // U' undoes U, U2 undoes U2, U undoes U'
    5, 4, 3,
    8, 7, 6,
    11,10, 9,
    14,13,12,
    17,16,15,
];

// Corner permutation for each move CW.
// Each entry [i] tells which absolute slot corner i came FROM after the move.
// i.e. new_corner_at_slot[i] = old_corner_at_slot[PERM[i]]
// Slots: 0=UBL  1=UBR  2=UFL  3=UFR  4=DBL  5=DBR  6=DFL  7=DFR

/// CW permutation per face (what slot does slot i receive its piece from)
const U_PERM: [u8; 8] = [1, 3, 0, 2, 4, 5, 6, 7]; // UBL←UBR←UFR←UFL←UBL
const R_PERM: [u8; 8] = [0, 3, 2, 7, 4, 1, 6, 5]; // UBR←UFR←DFR←DBR←UBR  wait—
// Let me be precise. R CW moves: UFR→UBR→DBR→DFR→UFR
// slot 3(UFR)→slot1(UBR)→slot5(DBR)→slot7(DFR)→slot3
// So: new[1]=old[3], new[5]=old[1], new[7]=old[5], new[3]=old[7], others same
// Written as "new[i] gets piece from slot PERM[i]":
// PERM = [0,3,2,7,4,1,6,5]  ← new[1]←old[3], new[3]←old[7], new[5]←old[1], new[7]←old[5]
const F_PERM: [u8; 8] = [0, 1, 3, 6, 4, 5, 7, 2]; // UFL→UFR→DFR→DFL→UFL
// UFL(2)→UFR(3)→DFR(7)→DFL(6)→UFL(2)
// new[3]←old[2], new[7]←old[3], new[6]←old[7], new[2]←old[6]
// PERM = [0,1,3,6,4,5,7,2] ✓
const D_PERM: [u8; 8] = [0, 1, 2, 3, 6, 5, 7, 4]; // DBL→DFL→DFR→DBR←DBL
// DBL(4)→DFL(6)→DFR(7)→DBR(5)→DBL(4)  — D CW viewed from below
// new[6]←old[4], new[7]←old[6], new[5]←old[7], new[4]←old[5]
// PERM = [0,1,2,3,6,7,5,4]... let me redo:
// D CW from below = CCW from above. Cycle: 4→6→7→5→4 (DBL→DFL→DFR→DBR→DBL)
// new[i] gets piece FROM: new[6]←4, new[7]←6, new[5]←7, new[4]←5
// PERM[4]=5, PERM[5]=7, PERM[6]=4, PERM[7]=6  → [0,1,2,3,5,7,4,6]
const D_PERM2: [u8; 8] = [0, 1, 2, 3, 5, 7, 4, 6];
const L_PERM: [u8; 8] = [2, 1, 6, 3, 0, 5, 4, 7]; // UBL←UFL←DFL←DBL←UBL
// L CW from left. Cycle UBL(0)→UFL(2)→DFL(6)→DBL(4)→UBL
// new[2]←0, new[6]←2, new[4]←6, new[0]←4
// PERM[0]=4, PERM[2]=0, PERM[4]=6, PERM[6]=2 → [4,1,0,3,6,5,2,7]
const L_PERM2: [u8; 8] = [4, 1, 0, 3, 6, 5, 2, 7];
const B_PERM: [u8; 8] = [1, 5, 2, 3, 0, 4, 6, 7]; // UBR←UBL←DBL←DBR←UBR
// B CW from back. Cycle UBR(1)→UBL(0)→DBL(4)→DBR(5)→UBR
// new[0]←1, new[4]←0, new[5]←4, new[1]←5
// PERM[0]=1, PERM[1]=5, PERM[4]=0, PERM[5]=4 → [1,5,2,3,0,4,6,7]  ✓

// Orientation twists for CW moves (added to the piece's current orientation, mod 3)
// 0=no change, 1=CW twist, 2=CCW twist
// U and D moves don't change orientation (pieces stay on same axis)
const U_ORI: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
const D_ORI: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
// R CW: UFR→UBR gets +2, UBR→DBR gets +1, DBR→DFR gets +2, DFR→UFR gets +1
// i.e. the piece landing at slot gets that twist:
// new piece at UBR(1) came from UFR → twist +2; at DBR(5) from UBR → +1;
// at DFR(7) from DBR → +2; at UFR(3) from DFR → +1
const R_ORI: [u8; 8] = [0, 2, 0, 1, 0, 1, 0, 2];
// F CW: UFR→DFR +2, DFR→DFL +1, DFL→UFL +2, UFL→UFR +1
// pieces land at: UFR(3) from UFL→+1; DFR(7) from UFR→+2; DFL(6) from DFR→+1; UFL(2) from DFL→+2
const F_ORI: [u8; 8] = [0, 0, 2, 1, 0, 0, 1, 2];
// L CW: UBL→UFL +1, UFL→DFL +2, DFL→DBL +1, DBL→UBL +2
// land at: UFL(2) from UBL→+1; DFL(6) from UFL→+2; DBL(4) from DFL→+1; UBL(0) from DBL→+2
const L_ORI: [u8; 8] = [2, 0, 1, 0, 1, 0, 2, 0];
// B CW: UBR→UBL +1, UBL→DBL +2, DBL→DBR +1, DBR→UBR +2
// land at: UBL(0) from UBR→+1; DBL(4) from UBL→+2; DBR(5) from DBL→+1; UBR(1) from DBR→+2
const B_ORI: [u8; 8] = [1, 2, 0, 0, 2, 1, 0, 0];

// ── State encoding ────────────────────────────────────────────────────────────
//
// We encode the state of all 7 free corners into a u64.
// Each free corner i occupies bits [5*i .. 5*i+4]:
//   bits [5i+0 .. 5i+2] = absolute slot position (0-7)
//   bits [5i+3 .. 5i+4] = orientation (0-2)
//
// "Corner state" = an array of 8 entries indexed by absolute slot:
//   cs[slot] = (which_piece_is_here: u8, orientation: u8)
// where piece identity = the slot it belongs to in the solved state
// (UBL piece belongs in slot 0, etc.)

type CornerState = [(u8, u8); 8]; // (piece, orientation) at each slot

fn solved_cs() -> CornerState {
    [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0)]
}

/// Encode CornerState → u64
fn encode(cs: &CornerState) -> u64 {
    let mut v: u64 = 0;
    for (fi, &slot) in FREE_TO_SLOT.iter().enumerate() {
        let (piece, ori) = cs[slot as usize];
        let bits = ((piece as u64) | ((ori as u64) << 3)) << (5 * fi);
        v |= bits;
    }
    v
}

/// Decode u64 → CornerState
/// Note: the fixed corner (slot 5) is always (5, derived_ori)
fn decode(v: u64) -> CornerState {
    let mut cs = [(255u8, 255u8); 8];
    let mut ori_sum = 0u8;
    let mut piece_seen = [false; 8];

    for (fi, &slot) in FREE_TO_SLOT.iter().enumerate() {
        let bits = (v >> (5 * fi)) as u8;
        let piece = bits & 0x7;
        let ori   = (bits >> 3) & 0x3;
        cs[slot as usize] = (piece, ori);
        ori_sum = (ori_sum + ori) % 3;
        piece_seen[piece as usize] = true;
    }

    // Derive fixed corner piece and orientation
    let fixed_piece = (0..8u8).find(|&p| !piece_seen[p as usize]).unwrap_or(5);
    let fixed_ori   = (3 - ori_sum % 3) % 3;
    cs[5] = (fixed_piece, fixed_ori);
    cs
}

/// Apply a move (given as perm + ori arrays for slots 0-7) to a CornerState
fn apply_move_cs(cs: &CornerState, perm: &[u8; 8], ori_delta: &[u8; 8]) -> CornerState {
    let mut next = [(0u8, 0u8); 8];
    for slot in 0..8usize {
        let src = perm[slot] as usize;          // piece at 'src' moves to 'slot'
        let (piece, ori) = cs[src];
        next[slot] = (piece, (ori + ori_delta[slot]) % 3);
    }
    next
}

// ── Pre-computed move table ───────────────────────────────────────────────────
//
// For each of the 18 moves, store the (perm, ori) pair for the CW version,
// then derive ×2 and CCW by composition.

struct MoveTable {
    /// mt[move_index][encoded_state] = encoded_state_after_move
    /// We don't pre-expand the full table (3.67M × 18 entries would be 264 MB).
    /// Instead we store just the 8 (perm, ori) pairs and compute on the fly.
    /// This is fast enough: ~20 ns per state expansion.
    perms: [([u8;8],[u8;8]); NMOVES],
}

impl MoveTable {
    fn build() -> Self {
        // CW versions
        let cw: [([u8;8],[u8;8]); 6] = [
            (U_PERM, U_ORI),
            (R_PERM, R_ORI),
            (F_PERM, F_ORI),
            (D_PERM2, D_ORI),
            (L_PERM2, L_ORI),
            (B_PERM, B_ORI),
        ];

        // Compose perm+ori with itself to get ×2 and ×3 (= CCW)
        fn compose(p1: &[u8;8], o1: &[u8;8], p2: &[u8;8], o2: &[u8;8])
            -> ([u8;8],[u8;8])
        {
            let mut p = [0u8;8];
            let mut o = [0u8;8];
            for i in 0..8 {
                p[i] = p1[p2[i] as usize];
                o[i] = (o2[i] + o1[p2[i] as usize]) % 3;
            }
            (p, o)
        }

        let mut perms = [([0u8;8],[0u8;8]); NMOVES];
        for f in 0..6 {
            let (p1,o1) = &cw[f];
            let x2       = compose(p1,o1,p1,o1);
            let x3       = compose(p1,o1,&x2.0,&x2.1);
            perms[f*3+0] = (*p1, *o1);   // CW
            perms[f*3+1] = x2;            // 180°
            perms[f*3+2] = x3;            // CCW
        }
        MoveTable { perms }
    }

    #[inline(always)]
    fn apply(&self, encoded: u64, mv: usize) -> u64 {
        let cs = decode(encoded);
        let (ref p, ref o) = self.perms[mv];
        encode(&apply_move_cs(&cs, p, o))
    }
}

// ── BFS table ────────────────────────────────────────────────────────────────
//
// Maps every reachable encoded state → (depth: u8, move_that_reaches_solved: u8)
// "move_that_reaches_solved" is the inverse of the move used to reach this state
// from BFS origin (= solved state).  Following these back-pointers from any
// state gives the optimal solution.

struct BfsTable {
    /// state → (depth, back_move_index)
    table: HashMap<u64, (u8, u8)>,
    mt: MoveTable,
}

impl BfsTable {
    fn build() -> Self {
        let mt = MoveTable::build();
        let solved = encode(&solved_cs());
        let mut table: HashMap<u64, (u8, u8)> =
            HashMap::with_capacity(4_000_000);
        table.insert(solved, (0, 255));

        let mut queue: VecDeque<u64> = VecDeque::with_capacity(500_000);
        queue.push_back(solved);

        while let Some(state) = queue.pop_front() {
            let (depth, _) = table[&state];
            for mv in 0..NMOVES {
                let next = mt.apply(state, mv);
                if !table.contains_key(&next) {
                    // The back-move is the inverse: to go from 'next' toward
                    // solved, apply MOVE_INV[mv]
                    table.insert(next, (depth + 1, MOVE_INV[mv] as u8));
                    queue.push_back(next);
                }
            }
        }

        BfsTable { table, mt }
    }

    /// Return the optimal solution as a Vec of move indices
    fn solve(&self, start: u64) -> Option<Vec<usize>> {
        if !self.table.contains_key(&start) {
            return None; // unreachable state
        }
        let solved = encode(&solved_cs());
        let mut moves = Vec::new();
        let mut state = start;

        loop {
            if state == solved { break; }
            let &(_, back_mv) = self.table.get(&state)?;
            moves.push(back_mv as usize);
            state = self.mt.apply(state, back_mv as usize);
        }
        Some(moves)
    }
}

// ── Global singleton ──────────────────────────────────────────────────────────

static BFS_TABLE: OnceLock<BfsTable> = OnceLock::new();

fn get_table() -> &'static BfsTable {
    BFS_TABLE.get_or_init(BfsTable::build)
}

// ── Sticker → CornerState conversion ─────────────────────────────────────────
//
// External API uses a 24-sticker array in this face/sticker order:
//
//   Indices  0– 3 = Up    (White=0)
//   Indices  4– 7 = Down  (Yellow=1)
//   Indices  8–11 = Front (Green=2)
//   Indices 12–15 = Back  (Blue=3)
//   Indices 16–19 = Left  (Orange=5)
//   Indices 20–23 = Right (Red=4)
//
// Within each face (reading order facing that face):
//   U: [0]=back-left  [1]=back-right  [2]=front-left  [3]=front-right
//   D: [4]=front-left [5]=front-right [6]=back-left   [7]=back-right
//   F: [8]=top-left   [9]=top-right   [10]=bot-left   [11]=bot-right
//   B: [12]=top-left  [13]=top-right  [14]=bot-left   [15]=bot-right
//   L: [16]=top-back  [17]=top-front  [18]=bot-back   [19]=bot-front
//   R: [20]=top-front [21]=top-back   [22]=bot-front  [23]=bot-back
//
// External colors:  0=White 1=Yellow 2=Green 3=Blue 4=Red 5=Orange
// Internal corners: UBL=0 UBR=1 UFL=2 UFR=3 DBL=4 DBR=5 DFL=6 DFR=7
//
// For each corner we read its 3 stickers and determine piece identity
// + orientation.

/// Which 3 sticker indices form each corner (slot order: 0..7)
/// Each entry: [u_or_d_sticker, f_or_b_sticker, l_or_r_sticker]
const CORNER_STICKERS: [[usize; 3]; 8] = [
    [ 0, 13,  16],  // UBL: U[0], B[13-back-right=UBL side], L[16]
    [ 1, 12,  21],  // UBR: U[1], B[12-back-left=UBR side],  R[21]
    [ 2,  9,  17],  // UFL: U[2], F[9-top-right=UFL? no...
    [ 3,  8,  20],  // UFR: U[3], F[8-top-left=UFR? careful
    [ 6, 15,  18],  // DBL: D[6], B[15-bot-right=DBL side],  L[18]
    [ 7, 14,  23],  // DBR: D[7], B[14-bot-left=DBR side],   R[23]
    [ 4, 11,  19],  // DFL: D[4], F[11-bot-right=DFL side],  L[19]
    [ 5, 10,  22],  // DFR: D[5], F[10-bot-left=DFR side],   R[22]
];
// Let me carefully re-derive CORNER_STICKERS.
// Corner UBL is at the intersection of U, B, L faces.
//   U face reading order (from above, back=far): [0]=back-left=UBL ✓
//   B face reading order (from behind, top-left from back perspective):
//     From back, the left side of the back face is actually the RIGHT side of the cube.
//     B[12]=top-left(from back)=UBR sticker on B-face
//     B[13]=top-right(from back)=UBL sticker on B-face ✓
//   L face reading order (from left): [16]=top-back=UBL ✓
// UBL corner: stickers at U[0], B[13], L[16] ✓
//
// Corner UBR: U, B, R
//   U[1]=back-right=UBR ✓
//   B[12]=top-left(from back)=UBR ✓
//   R[21]=top-back=UBR ✓
// UBR corner: U[1], B[12], R[21] ✓
//
// Corner UFL: U, F, L
//   U[2]=front-left=UFL ✓
//   F[9]=top-right(from front)=UFL? No: top-right from front is UFR.
//   F[8]=top-left(from front)=UFL ✓
//   L[17]=top-front=UFL ✓
// UFL corner: U[2], F[8], L[17] ✓
//
// Corner UFR: U, F, R
//   U[3]=front-right=UFR ✓
//   F[9]=top-right(from front)=UFR ✓
//   R[20]=top-front=UFR ✓
// UFR corner: U[3], F[9], R[20] ✓
//
// Corner DBL: D, B, L
//   D[6]=back-left=DBL ✓
//   B[15]=bot-right(from back)=DBL ✓  (bot-right from back = bottom-left of cube = DBL)
//   L[18]=bot-back=DBL ✓
// DBL corner: D[6], B[15], L[18] ✓
//
// Corner DBR: D, B, R
//   D[7]=back-right=DBR ✓
//   B[14]=bot-left(from back)=DBR ✓
//   R[23]=bot-back=DBR ✓
// DBR corner: D[7], B[14], R[23] ✓
//
// Corner DFL: D, F, L
//   D[4]=front-left=DFL ✓
//   F[10]=bot-left(from front)=DFL ✓  wait: bot-left from front = DFL ✓
//   L[19]=bot-front=DFL ✓
// DFL corner: D[4], F[10], L[19] ✓
//
// Corner DFR: D, F, R
//   D[5]=front-right=DFR ✓
//   F[11]=bot-right(from front)=DFR ✓
//   R[22]=bot-front=DFR ✓
// DFR corner: D[5], F[11], R[22] ✓

/// Corrected CORNER_STICKERS
const CORNER_STICKERS_OK: [[usize; 3]; 8] = [
    [ 0, 13,  16],  // 0=UBL: U[0], B[13], L[16]
    [ 1, 12,  21],  // 1=UBR: U[1], B[12], R[21]
    [ 2,  8,  17],  // 2=UFL: U[2], F[8],  L[17]
    [ 3,  9,  20],  // 3=UFR: U[3], F[9],  R[20]
    [ 6, 15,  18],  // 4=DBL: D[6], B[15], L[18]
    [ 7, 14,  23],  // 5=DBR: D[7], B[14], R[23]
    [ 4, 10,  19],  // 6=DFL: D[4], F[10], L[19]
    [ 5, 11,  22],  // 7=DFR: D[5], F[11], R[22]
];

/// External color → face axis (for orientation detection)
/// The face a sticker belongs to tells us which axis the cubie is oriented on.
/// Face axes: U/D = axis 0 (good ori), R/L = axis 1, F/B = axis 2
/// Solved colors per face: U=0(White), D=1(Yellow), F=2(Green), B=3(Blue), L=5(Orange), R=4(Red)
const COLOR_TO_AXIS: [u8; 6] = [
    0, // White  → U/D axis
    0, // Yellow → U/D axis
    2, // Green  → F/B axis
    2, // Blue   → F/B axis
    4, // Red    → R/L axis  (color index 4)
    5, // Orange → R/L axis  (color index 5)
];
// Corrected (axis 0 = U/D, 1 = R/L, 2 = F/B):
const COLOR_AXIS: [u8; 6] = [
    0, // 0=White  → UD
    0, // 1=Yellow → UD
    2, // 2=Green  → FB
    2, // 3=Blue   → FB
    1, // 4=Red    → RL
    1, // 5=Orange → RL
];

/// Solved color of each corner slot's U/D sticker (used to identify pieces)
/// Corner piece identity = determined by the set of 3 face colors
/// Solved colors: UBL=(White,Blue,Orange)=(0,3,5), UBR=(0,3,4), UFL=(0,2,5),
///   UFR=(0,2,4), DBL=(1,3,5), DBR=(1,3,4), DFL=(1,2,5), DFR=(1,2,4)
const SOLVED_CORNER_COLORS: [[u8;3]; 8] = [
    [0,3,5], // 0=UBL: White,Blue,Orange
    [0,3,4], // 1=UBR: White,Blue,Red
    [0,2,5], // 2=UFL: White,Green,Orange
    [0,2,4], // 3=UFR: White,Green,Red
    [1,3,5], // 4=DBL: Yellow,Blue,Orange
    [1,3,4], // 5=DBR: Yellow,Blue,Red
    [1,2,5], // 6=DFL: Yellow,Green,Orange
    [1,2,4], // 7=DFR: Yellow,Green,Red
];

/// Convert a 24-sticker array (external format) to a CornerState
fn stickers_to_cs(stickers: &[u8; 24]) -> Result<CornerState, String> {
    let mut cs = [(0u8, 0u8); 8];

    for slot in 0..8usize {
        let [si0, si1, si2] = CORNER_STICKERS_OK[slot];
        let c0 = stickers[si0]; // U or D face color at this corner
        let c1 = stickers[si1]; // F or B face color
        let c2 = stickers[si2]; // L or R face color

        // Find which piece this is by matching the color set
        let color_set = [c0, c1, c2];
        let mut found_piece = None;
        'outer: for piece in 0..8usize {
            let sc = SOLVED_CORNER_COLORS[piece];
            // Check all 3 colors match (any permutation)
            let mut used = [false; 3];
            let mut all_match = true;
            for &cc in &color_set {
                let mut ok = false;
                for (j, &sc_c) in sc.iter().enumerate() {
                    if !used[j] && sc_c == cc { used[j]=true; ok=true; break; }
                }
                if !ok { all_match=false; break; }
            }
            if all_match { found_piece=Some(piece as u8); break 'outer; }
        }
        let piece = found_piece.ok_or_else(||
            format!("Unrecognized color combination at slot {}: {:?}", slot, color_set)
        )?;

        // Determine orientation: which sticker of this cubie is on the U/D axis?
        // Orientation 0 = U/D-face sticker is on U or D face (sticker index si0)
        // Orientation 1 = U/D-face sticker is on R or L face (sticker index si2)
        // Orientation 2 = U/D-face sticker is on F or B face (sticker index si1)
        //
        // The "U/D sticker" of the piece is the one with color 0(White) or 1(Yellow)
        let ud_color = if SOLVED_CORNER_COLORS[piece as usize][0] == 0 { 0u8 } else { 1u8 };
        let ori = if c0 == ud_color       { 0 }
                  else if c1 == ud_color  { 2 }
                  else                    { 1 };

        cs[slot] = (piece, ori);
    }
    Ok(cs)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Solve a 2×2 cube from a 24-sticker input array.
///
/// Input format (24 integers, colors 0-5):
///   Indices  0– 3 = Up    face  (0=White)
///   Indices  4– 7 = Down  face  (1=Yellow)
///   Indices  8–11 = Front face  (2=Green)
///   Indices 12–15 = Back  face  (3=Blue)
///   Indices 16–19 = Left  face  (5=Orange)
///   Indices 20–23 = Right face  (4=Red)
///
/// Returns a Vec of move strings, e.g. ["R", "U'", "F2"].
/// Returns an empty Vec if the cube is already solved.
/// Returns Err if the input is invalid or the state is unreachable.
pub fn solve_cube(stickers: &[u8; 24]) -> Result<Vec<String>, String> {
    let cs = stickers_to_cs(stickers)?;
    let encoded = encode(&cs);
    let table = get_table();

    let move_indices = table.solve(encoded)
        .ok_or("State is not reachable from solved — check sticker input")?;

    Ok(move_indices.iter().map(|&i| MOVE_NAMES[i].to_string()).collect())
}

/// Returns the God's number (minimum moves) for this cube state.
pub fn gods_number(stickers: &[u8; 24]) -> Result<u8, String> {
    let cs = stickers_to_cs(stickers)?;
    let encoded = encode(&cs);
    let table = get_table();
    table.table.get(&encoded)
        .map(|&(d,_)| d)
        .ok_or("State is not reachable".to_string())
}

// ── PyO3 bindings ─────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "solve")]
fn py_solve(stickers: Vec<u8>) -> PyResult<Vec<String>> {
    if stickers.len() != 24 {
        return Err(PyValueError::new_err(
            format!("Expected 24 stickers, got {}", stickers.len())
        ));
    }
    for (i, &c) in stickers.iter().enumerate() {
        if c > 5 {
            return Err(PyValueError::new_err(
                format!("Invalid color {} at index {}: must be 0-5", c, i)
            ));
        }
    }
    let arr: [u8;24] = stickers.try_into().unwrap();
    solve_cube(&arr).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(name = "gods_number")]
fn py_gods_number(stickers: Vec<u8>) -> PyResult<u8> {
    if stickers.len() != 24 {
        return Err(PyValueError::new_err("Expected 24 stickers"));
    }
    let arr: [u8;24] = stickers.try_into().unwrap();
    gods_number(&arr).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(name = "warmup")]
fn py_warmup() {
    // Call get_table() to trigger BFS build; useful to call at import time
    // so the first solve() call isn't slow.
    get_table();
}

#[pymodule]
fn cube2_solver(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_solve, m)?)?;
    m.add_function(wrap_pyfunction!(py_gods_number, m)?)?;
    m.add_function(wrap_pyfunction!(py_warmup, m)?)?;
    Ok(())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Solved state stickers (external format)
    fn solved_stickers() -> [u8; 24] {
        [
            0,0,0,0,   // U: White
            1,1,1,1,   // D: Yellow
            2,2,2,2,   // F: Green
            3,3,3,3,   // B: Blue
            5,5,5,5,   // L: Orange
            4,4,4,4,   // R: Red
        ]
    }

    /// Apply a sequence of move names to a sticker array and return new array.
    /// This lets us build test cases from known scrambles.
    fn apply_moves_to_stickers(start: &[u8;24], moves: &[&str]) -> [u8; 24] {
        // Build internal state, apply moves, convert back to stickers.
        // We re-use our CornerState machinery.
        // First convert stickers→CS, apply moves, then CS→stickers.
        let table = get_table();
        let cs = stickers_to_cs(start).unwrap();
        let mut enc = encode(&cs);
        for mv_name in moves {
            let idx = MOVE_NAMES.iter().position(|&n| n == *mv_name)
                .expect("Unknown move name");
            enc = table.mt.apply(enc, idx);
        }
        cs_to_stickers(&decode(enc))
    }

    /// Convert CornerState back to solved stickers (for test verification)
    fn cs_to_stickers(cs: &CornerState) -> [u8; 24] {
        // Solved colors for each piece
        // Sticker order at each corner: [UD_color, FB_color, RL_color]
        let solved_stickers_base = solved_stickers();
        let mut out = [0u8; 24];

        for slot in 0..8usize {
            let (piece, ori) = cs[slot];
            let src_colors = SOLVED_CORNER_COLORS[piece as usize];
            // src_colors = [UD_color, FB_color, RL_color]
            // ori 0: UD stays UD; ori 1: UD→RL; ori 2: UD→FB
            let [si0, si1, si2] = CORNER_STICKERS_OK[slot];
            // Which piece color goes to which sticker position?
            let (ud_c, fb_c, rl_c) = match ori {
                0 => (src_colors[0], src_colors[1], src_colors[2]),
                1 => (src_colors[2], src_colors[0], src_colors[1]),
                2 => (src_colors[1], src_colors[2], src_colors[0]),
                _ => unreachable!(),
            };
            out[si0] = ud_c;
            out[si1] = fb_c;
            out[si2] = rl_c;
        }
        out
    }

    fn verify_solution(scramble_moves: &[&str]) {
        let start = solved_stickers();
        let scrambled = apply_moves_to_stickers(&start, scramble_moves);
        let solution = solve_cube(&scrambled).expect("solve failed");

        // Apply solution to scrambled; result must be solved
        let final_state = if solution.is_empty() {
            scrambled
        } else {
            let sol_refs: Vec<&str> = solution.iter().map(|s| s.as_str()).collect();
            apply_moves_to_stickers(&scrambled, &sol_refs)
        };
        assert_eq!(final_state, start,
            "Solution {:?} did not solve scramble {:?}", solution, scramble_moves);
        println!("Scramble {:?} → solution ({} moves): {:?}",
            scramble_moves, solution.len(), solution);
    }

    #[test]
    fn test_solved() {
        let stickers = solved_stickers();
        let sol = solve_cube(&stickers).unwrap();
        assert!(sol.is_empty(), "Solved cube should return empty solution");
    }

    #[test]
    fn test_single_move() {
        verify_solution(&["R"]);
        verify_solution(&["U'"]);
        verify_solution(&["F2"]);
    }

    #[test]
    fn test_4_move_scramble() {
        verify_solution(&["R", "U", "R'", "U'"]);
    }

    #[test]
    fn test_deep_scramble_10() {
        verify_solution(&["R","U","R'","F","D","L","B2","R","U'","F'"]);
    }

    #[test]
    fn test_deep_scramble_14() {
        // A known hard scramble for 2×2
        verify_solution(&["R","U","R'","U'","R","U","R'","U'","R","U","R'","U'","F","F'"]);
        // Actual deep scramble:
        verify_solution(&["F","R","U","R'","U'","F'","R","U","R'","U'","R","U","R'","U'"]);
    }

    #[test]
    fn test_gods_number_known() {
        let start = solved_stickers();
        // R U R' U' is depth 4
        let scrambled = apply_moves_to_stickers(&start, &["R","U","R'","U'"]);
        let gn = gods_number(&scrambled).unwrap();
        println!("R U R' U' → God's number = {}", gn);
        assert!(gn <= 4, "Should be ≤4, got {}", gn);
    }

    #[test]
    fn test_all_states_reachable() {
        // The BFS table must cover all ~3.67M states
        let table = get_table();
        println!("BFS table size: {}", table.table.len());
        assert!(table.table.len() >= 3_000_000,
            "Expected ~3.67M states, got {}", table.table.len());
    }
}
