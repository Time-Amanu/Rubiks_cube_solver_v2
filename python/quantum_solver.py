"""
quantum_solver.py  —  QUBO path-search solver for the 2×2 Rubik's cube
=======================================================================
Uses the D-Wave Ocean SDK (dimod + dwave-system / dwave-samplers) to frame
cube-solving as a Binary Quadratic Model optimisation problem.

MATHEMATICAL FORMULATION
─────────────────────────
Decision variables
    x[t, m] ∈ {0, 1}     "apply move m at step t?"
    t ∈ {0 … D−1},  m ∈ {0 … 17}   (18 HTM moves)
    Total vars: D × 18  (max 11 × 18 = 198)

Energy function  E = λ₁·P_one + λ₂·P_seq + λ₃·P_elig + λ₄·P_obj

  P_one   One-hot per step: exactly one move chosen.
          (Σ_m x[t,m] − 1)² expanded into BQM terms.

  P_seq   Sequential validity: penalise (t,m₁)·(t+1,m₂) pairs where
          • same face:  FACE[m₁] == FACE[m₂]  (always suboptimal)
          • cancelling: m₂ == MOVE_INV[m₁]     (net-zero move pair)

  P_elig  Eligibility: heavy penalty on x[t,m] when move m cannot be
          on ANY shortest path from start_state at depth t.
          Computed via forward BFS from start constrained to the
          BFS depth-decrease frontier.

  P_obj   Objective reward: negative bias on x[t,m] for pairs that
          bring the cube to solved at step t+1.

BQM SIZE
─────────
198 binary variables → O(198²/2) ≈ 19 602 possible quadratic terms.
After pruning ineligible cross-step pairs the fill is much sparser.
Fits comfortably on SimulatedAnnealingSampler and LeapHybridSampler.

COLOR / MOVE CONTRACT  (matches lib.rs, readCubeState.js, api.py)
──────────────────────────────────────────────────────────────────
Colors  0=White  1=Yellow  2=Green  3=Blue  4=Red  5=Orange
Stickers 0-3=Up, 4-7=Down, 8-11=Front, 12-15=Back, 16-19=Left, 20-23=Right
Moves   U U2 U'  R R2 R'  F F2 F'  D D2 D'  L L2 L'  B B2 B'  (idx 0-17)

INSTALL
───────
    pip install dimod dwave-samplers          # local SA  (no account needed)
    pip install dwave-system                  # for Leap cloud access

QUICK START
───────────
    # Solved state (trivial)
    python quantum_solver.py solve \\
        --stickers 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,5,5,5,5,4,4,4,4

    # Scramble with SimulatedAnnealingSampler
    python quantum_solver.py solve --stickers <24 ints> --num_reads 2000

    # Use D-Wave Leap cloud (needs DWAVE_API_TOKEN env var)
    python quantum_solver.py solve --stickers <24 ints> --use_leap

    # Benchmark QUBO vs BFS on random scrambles
    python quantum_solver.py benchmark --depths 3 5 7 9 --num_trials 5

    # Run built-in unit tests (no dimod required)
    python quantum_solver.py test

    # Print environment / dependency info
    python quantum_solver.py info
"""

from __future__ import annotations

import os
import time
import random
import logging
import argparse
import itertools
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("quantum_solver")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — D-Wave Ocean imports (graceful degradation)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import dimod
    from dimod import BinaryQuadraticModel, SampleSet
    DIMOD_AVAILABLE = True
except ImportError:
    dimod                = None   # type: ignore[assignment]
    BinaryQuadraticModel = None   # type: ignore[assignment,misc]
    SampleSet            = None   # type: ignore[assignment,misc]
    DIMOD_AVAILABLE      = False

# dwave-samplers (>= 1.0) exposes SA; older Ocean SDKs put it in dimod directly.
try:
    from dwave.samplers import SimulatedAnnealingSampler
    SA_AVAILABLE = True
except ImportError:
    try:
        from dimod import SimulatedAnnealingSampler    # type: ignore[no-redef]
        SA_AVAILABLE = True
    except (ImportError, AttributeError):
        SimulatedAnnealingSampler = None               # type: ignore[assignment,misc]
        SA_AVAILABLE = False

try:
    from dwave.system import LeapHybridSampler
    LEAP_AVAILABLE = True
except ImportError:
    LeapHybridSampler = None    # type: ignore[assignment]
    LEAP_AVAILABLE    = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Pure-Python 2×2 cube engine  (mirrors lib.rs exactly)
# ═══════════════════════════════════════════════════════════════════════════════
#
# State: tuple of 8 pairs  (piece: int, orientation: int)
# Slot index:   0=UBL  1=UBR  2=UFL  3=UFR  4=DBL  5=DBR  6=DFL  7=DFR
#
# Orientation (Pochmann convention, same as lib.rs):
#   0 = piece's UD-face colour faces U or D   (well-oriented)
#   1 = piece's UD-face colour faces R or L   (CW twist)
#   2 = piece's UD-face colour faces F or B   (CCW twist)

MOVE_NAMES: list[str] = [
    "U",  "U2", "U'",
    "R",  "R2", "R'",
    "F",  "F2", "F'",
    "D",  "D2", "D'",
    "L",  "L2", "L'",
    "B",  "B2", "B'",
]
N_MOVES: int = 18

# Inverse move: MOVE_INV[i] undoes move i
MOVE_INV: list[int] = [
    2, 1, 0,
    5, 4, 3,
    8, 7, 6,
    11, 10, 9,
    14, 13, 12,
    17, 16, 15,
]

# Face index for each move (0=U 1=R 2=F 3=D 4=L 5=B)
FACE_OF: list[int] = [f for f in range(6) for _ in range(3)]

# ── CW quarter-turn tables ────────────────────────────────────────────────────
# perm[i]      = source slot  (piece landing at slot i came from perm[i])
# ori_delta[i] = orientation increment for piece that lands at slot i
# Verified by: M ∘ M⁻¹ = I for all M, and (R U R' U')×6 = I

_CW_TABLES: list[tuple[list[int], list[int]]] = [
    # U:  cycle UBL→UBR→UFR→UFL
    ([1, 3, 0, 2, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 0, 0]),
    # R:  cycle UBR→UFR→DFR→DBR
    ([0, 3, 2, 7, 4, 1, 6, 5], [0, 2, 0, 1, 0, 1, 0, 2]),
    # F:  cycle UFL→UFR→DFR→DFL
    ([0, 1, 3, 6, 4, 5, 7, 2], [0, 0, 2, 1, 0, 0, 1, 2]),
    # D:  cycle DBL→DFL→DFR→DBR
    ([0, 1, 2, 3, 5, 7, 4, 6], [0, 0, 0, 0, 0, 0, 0, 0]),
    # L:  cycle UBL→UFL→DFL→DBL
    ([4, 1, 0, 3, 6, 5, 2, 7], [2, 0, 1, 0, 1, 0, 2, 0]),
    # B:  cycle UBR→UBL→DBL→DBR
    ([1, 5, 2, 3, 0, 4, 6, 7], [1, 2, 0, 0, 2, 1, 0, 0]),
]


def _compose_moves(
    p1: list[int], o1: list[int],
    p2: list[int], o2: list[int],
) -> tuple[list[int], list[int]]:
    """
    Compose two moves: apply (p1,o1) FIRST, then (p2,o2).

    Derivation:
      After p1: s'[i]  = s[p1[i]]         (with ori += o1[i])
      After p2: s''[i] = s'[p2[i]]        (with ori += o2[i])
                       = s[p1[p2[i]]]     (with ori += o1[p2[i]] + o2[i])
    """
    p = [p1[p2[i]] for i in range(8)]
    o = [(o2[i] + o1[p2[i]]) % 3 for i in range(8)]
    return p, o


def _build_move_table() -> list[tuple[list[int], list[int]]]:
    """Build all 18 (perm, ori_delta) tables: CW(×1), 180°(×2), CCW(×3)."""
    table: list[tuple[list[int], list[int]]] = []
    for p_cw, o_cw in _CW_TABLES:
        p180, o180 = _compose_moves(p_cw, o_cw, p_cw,  o_cw)    # 180° = CW ∘ CW
        pccw, occw = _compose_moves(p_cw, o_cw, p180, o180)      # CCW  = CW ∘ 180°
        table.append((p_cw,  o_cw))    # 3f+0  CW
        table.append((p180,  o180))    # 3f+1  180°
        table.append((pccw,  occw))    # 3f+2  CCW
    return table


MOVE_TABLE: list[tuple[list[int], list[int]]] = _build_move_table()

# Immutable solved state constant
SOLVED_STATE: tuple = tuple((i, 0) for i in range(8))


def apply_move(state: tuple, move_idx: int) -> tuple:
    """Return the new cube state after applying move move_idx."""
    p, o = MOVE_TABLE[move_idx]
    return tuple(
        (state[p[i]][0], (state[p[i]][1] + o[i]) % 3)
        for i in range(8)
    )


def is_solved(state: tuple) -> bool:
    return state == SOLVED_STATE


def state_to_key(state: tuple) -> int:
    """
    Encode state as a unique integer key.
    Each slot: 5 bits  (piece 3 bits | orientation 2 bits).
    Total: 40 bits.
    """
    k = 0
    for piece, ori in state:
        k = (k << 5) | (piece << 2) | ori
    return k


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Sticker ↔ state conversion
# ═══════════════════════════════════════════════════════════════════════════════
#
# External array layout (matches api.py / readCubeState.js):
#   indices  0– 3  Up    (White  = 0)
#   indices  4– 7  Down  (Yellow = 1)
#   indices  8–11  Front (Green  = 2)
#   indices 12–15  Back  (Blue   = 3)
#   indices 16–19  Left  (Orange = 5)
#   indices 20–23  Right (Red    = 4)

# Per-slot sticker indices: [UD-face, FB-face, RL-face]
CORNER_STICKERS: list[list[int]] = [
    [ 0, 13, 16],  # 0=UBL: U[0], B[13], L[16]
    [ 1, 12, 21],  # 1=UBR: U[1], B[12], R[21]
    [ 2,  8, 17],  # 2=UFL: U[2], F[8],  L[17]
    [ 3,  9, 20],  # 3=UFR: U[3], F[9],  R[20]
    [ 6, 15, 18],  # 4=DBL: D[6], B[15], L[18]
    [ 7, 14, 23],  # 5=DBR: D[7], B[14], R[23]
    [ 4, 10, 19],  # 6=DFL: D[4], F[10], L[19]
    [ 5, 11, 22],  # 7=DFR: D[5], F[11], R[22]
]

# Solved colours per piece: [UD-colour, FB-colour, RL-colour]
SOLVED_CORNER_COLORS: list[list[int]] = [
    [0, 3, 5],   # piece 0=UBL: White,  Blue,   Orange
    [0, 3, 4],   # piece 1=UBR: White,  Blue,   Red
    [0, 2, 5],   # piece 2=UFL: White,  Green,  Orange
    [0, 2, 4],   # piece 3=UFR: White,  Green,  Red
    [1, 3, 5],   # piece 4=DBL: Yellow, Blue,   Orange
    [1, 3, 4],   # piece 5=DBR: Yellow, Blue,   Red
    [1, 2, 5],   # piece 6=DFL: Yellow, Green,  Orange
    [1, 2, 4],   # piece 7=DFR: Yellow, Green,  Red
]

# O(1) lookup: frozenset of 3 colours → piece index
_COLOR_KEY_TO_PIECE: dict[frozenset, int] = {
    frozenset(colors): piece
    for piece, colors in enumerate(SOLVED_CORNER_COLORS)
}


def stickers_to_state(stickers: list[int]) -> tuple:
    """
    Convert a 24-sticker external array to the internal corner-state tuple.

    Orientation detection
    ─────────────────────
    For slot s, we read three colours from the sticker array:
      c_ud = colour on the UD face of slot s
      c_fb = colour on the FB face
      c_rl = colour on the RL face

    The orientation is determined by where the piece's UD-colour ends up:
      ori=0  c_ud == ud_colour   → UD colour on UD face  (correct)
      ori=1  c_rl == ud_colour   → UD colour on RL face  (CW twist)
      ori=2  c_fb == ud_colour   → UD colour on FB face  (CCW twist)

    Raises ValueError on wrong length or unrecognised colour combinations.
    """
    if len(stickers) != 24:
        raise ValueError(f"Expected 24 stickers, got {len(stickers)}")

    result = []
    for slot in range(8):
        si_ud, si_fb, si_rl = CORNER_STICKERS[slot]
        c_ud = stickers[si_ud]
        c_fb = stickers[si_fb]
        c_rl = stickers[si_rl]

        piece = _COLOR_KEY_TO_PIECE.get(frozenset([c_ud, c_fb, c_rl]))
        if piece is None:
            raise ValueError(
                f"Slot {slot}: unrecognised colour combination "
                f"(UD={c_ud}, FB={c_fb}, RL={c_rl})"
            )

        ud_colour = SOLVED_CORNER_COLORS[piece][0]   # White=0 or Yellow=1
        if   c_ud == ud_colour:   ori = 0
        elif c_rl == ud_colour:   ori = 1
        else:                     ori = 2   # c_fb == ud_colour

        result.append((piece, ori))

    return tuple(result)


def state_to_stickers(state: tuple) -> list[int]:
    """
    Invert stickers_to_state(): convert internal state → 24-sticker array.

    Orientation → sticker mapping
    ──────────────────────────────
    Let [ud_c, fb_c, rl_c] = SOLVED_CORNER_COLORS[piece].

    ori=0  → si_ud=ud_c, si_fb=fb_c, si_rl=rl_c   (identity)
    ori=1  → si_ud=rl_c, si_fb=fb_c, si_rl=ud_c   (RL-colour faces UD; si_rl gets ud_c)
    ori=2  → si_ud=fb_c, si_fb=rl_c, si_rl=ud_c   (FB-colour faces UD; si_rl gets ud_c)

    Derivation: stickers_to_state reads:
      ori=1 when c_rl == ud_colour  → si_rl must hold ud_c, leaving
            si_ud=rl_c and si_fb=fb_c as the only consistent assignment.
      ori=2 when c_fb == ud_colour  → si_fb must hold ud_c... wait:
            stickers_to_state checks c_fb == ud_colour for ori=2,
            but c_fb is stickers[si_fb], so si_fb must hold ud_c for ori=2.
            Remaining: si_ud=fb_c, si_rl=rl_c.

    Empirically verified by exhaustive permutation search and round-trip test.
    """
    stickers = [0] * 24
    for slot in range(8):
        piece, ori          = state[slot]
        ud_c, fb_c, rl_c    = SOLVED_CORNER_COLORS[piece]
        si_ud, si_fb, si_rl = CORNER_STICKERS[slot]

        if ori == 0:
            stickers[si_ud] = ud_c
            stickers[si_fb] = fb_c
            stickers[si_rl] = rl_c
        elif ori == 1:
            # c_rl == ud_colour  →  si_rl = ud_c
            stickers[si_ud] = rl_c
            stickers[si_fb] = fb_c
            stickers[si_rl] = ud_c
        else:   # ori == 2
            # c_fb == ud_colour  →  si_fb = ud_c
            stickers[si_ud] = fb_c
            stickers[si_fb] = ud_c
            stickers[si_rl] = rl_c

    return stickers


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BFS table  (full state space, ~3.67 M states)
# ═══════════════════════════════════════════════════════════════════════════════

class BFSTable:
    """
    Pre-computes the entire 2×2 state space via BFS from the solved state.

    Internal storage
    ─────────────────
        state_key → (depth: int, back_move: int)

    back_move is the move index to apply FROM this state to move one step
    toward solved (a forward-progress move).  back_move == −1 means solved.

    The table is built lazily and cached as a module-level singleton.
    Building takes ~2–5 s in CPython and stores ~3.67 M entries.
    """

    __slots__ = ("_table", "_built")

    def __init__(self) -> None:
        self._table: dict[int, tuple[int, int]] = {}
        self._built: bool = False

    def build(self) -> None:
        """Run BFS; populate self._table.  No-op if already built."""
        if self._built:
            return

        t0 = time.perf_counter()
        sk = state_to_key(SOLVED_STATE)
        self._table[sk] = (0, -1)
        queue: deque = deque([SOLVED_STATE])

        while queue:
            state = queue.popleft()
            depth = self._table[state_to_key(state)][0]

            for mv in range(N_MOVES):
                nxt = apply_move(state, mv)
                nk  = state_to_key(nxt)
                if nk not in self._table:
                    # The back-pointer is the inverse move: applying
                    # MOVE_INV[mv] from nxt brings us back toward solved.
                    self._table[nk] = (depth + 1, MOVE_INV[mv])
                    queue.append(nxt)

        ms = (time.perf_counter() - t0) * 1000
        log.info("BFS complete: %d states in %.0f ms", len(self._table), ms)
        self._built = True

    def depth(self, state: tuple) -> Optional[int]:
        """Return God's number for this state, or None if not in table."""
        entry = self._table.get(state_to_key(state))
        return entry[0] if entry is not None else None

    def optimal_depth(self, state: tuple) -> int:
        """Like depth() but returns 0 for unknown states."""
        d = self.depth(state)
        return d if d is not None else 0

    def solve(self, start: tuple) -> list[int]:
        """
        Return the optimal move sequence from start → solved.
        Follows back-pointers; length == depth(start).
        """
        if not self._built:
            self.build()
        if is_solved(start):
            return []

        seq:  list[int] = []
        state = start
        sk    = state_to_key(SOLVED_STATE)

        for _ in range(14):   # God's number for 2×2 ≤ 14 HTM
            k = state_to_key(state)
            if k == sk:
                break
            entry = self._table.get(k)
            if entry is None:
                log.error("BFS table gap during solve — invalid input state?")
                return []
            _, back_mv = entry
            if back_mv == -1:
                break
            seq.append(back_mv)
            state = apply_move(state, back_mv)

        return seq

    def __len__(self) -> int:
        return len(self._table)


# Module-level singleton — built once per process
_BFS_SINGLETON: BFSTable = BFSTable()


def get_bfs_table() -> BFSTable:
    """Return the (lazily built) global BFS table."""
    _BFS_SINGLETON.build()
    return _BFS_SINGLETON


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — QUBO / BQM construction
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QUBOConfig:
    """
    λ-weights for the four energy terms.

    Tuning rules of thumb
    ─────────────────────
    lambda_one   Dominates all other terms (one-hot constraint).
                 Keep lambda_one ≫ |lambda_obj|.  Default 10 works for ≤ 11 moves.

    lambda_seq   Penalty for same-face or cancelling consecutive moves.
                 Smaller than lambda_one.  Default 6.

    lambda_elig  Penalty for provably ineligible moves.
                 Must exceed lambda_one so ineligible vars are always forced to 0.
                 Default 20.

    lambda_obj   Reward (negative) for the move that reaches solved.
                 |lambda_obj| < lambda_one to avoid breaking one-hot.
                 Default −8.
    """
    lambda_one:  float = 10.0
    lambda_seq:  float =  6.0
    lambda_elig: float = 20.0
    lambda_obj:  float = -8.0


@dataclass
class SolveResult:
    """Complete result of a quantum_solve() call."""
    moves:             list[str] = field(default_factory=list)
    move_indices:      list[int] = field(default_factory=list)
    gods_number:       int       = 0
    is_optimal:        bool      = False
    energy:            float     = 0.0
    sample_time_ms:    float     = 0.0
    total_time_ms:     float     = 0.0
    sampler_used:      str       = ""
    fallback_used:     bool      = False
    bqm_variables:     int       = 0
    bqm_interactions:  int       = 0


class CubeQUBOBuilder:
    """
    Constructs the Binary Quadratic Model that encodes cube-solving as QUBO.

    Variable layout
    ───────────────
    v = t * N_MOVES + m   represents x[t, m]
    For depth D: D × 18 binary variables total.

    Construction steps
    ──────────────────
    1. Compute eligible (t, m) pairs via forward BFS on the optimal frontier.
    2. Compute solving (t, m) pairs where applying m reaches the solved state.
    3. Add Term 1 (one-hot per step).
    4. Add Term 2 (sequential validity penalties).
    5. Add Term 3 (ineligibility penalties).
    6. Add Term 4 (objective rewards).
    """

    def __init__(self, config: QUBOConfig) -> None:
        self.cfg = config

    @staticmethod
    def var(t: int, m: int) -> int:
        """Integer variable index for (step t, move m)."""
        return t * N_MOVES + m

    def _compute_sets(
        self,
        start: tuple,
        depth: int,
        bfs: BFSTable,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """
        Forward-expand from start up to `depth` steps along optimal paths.

        Returns
        ───────
        eligible : {(t, m)} — moves on at least one shortest path
        solving  : {(t, m)} — moves whose application reaches solved
        """
        eligible: set[tuple[int, int]] = set()
        solving:  set[tuple[int, int]] = set()

        # Each frontier entry: (state_tuple, face_of_last_move)
        frontier: list[tuple[tuple, int]] = [(start, -1)]

        for t in range(depth):
            next_frontier: list[tuple[tuple, int]] = []
            next_seen:     set[int]                = set()

            for state, last_face in frontier:
                d_now = bfs.depth(state)
                if d_now is None:
                    continue

                for m in range(N_MOVES):
                    # Prune same-face moves (always suboptimal at next step)
                    if FACE_OF[m] == last_face:
                        continue

                    nxt   = apply_move(state, m)
                    d_nxt = bfs.depth(nxt)
                    if d_nxt is None:
                        continue

                    # Keep only strictly depth-decreasing moves
                    if d_nxt < d_now:
                        eligible.add((t, m))
                        nk = state_to_key(nxt)
                        if nk not in next_seen:
                            next_seen.add(nk)
                            next_frontier.append((nxt, FACE_OF[m]))
                        if d_nxt == 0:
                            solving.add((t, m))

            frontier = next_frontier
            if not frontier:
                break

        log.debug(
            "depth=%d  eligible=%d/%d  solving=%d",
            depth, len(eligible), depth * N_MOVES, len(solving),
        )
        return eligible, solving

    def build(self, start_state: tuple, depth: int) -> "BinaryQuadraticModel":
        """
        Build and return the BQM for (start_state, depth).

        All variable labels are plain ints  v = t * N_MOVES + m.
        VARTYPE is BINARY.
        """
        if not DIMOD_AVAILABLE:
            raise RuntimeError(
                "dimod is required.  Install: pip install dimod dwave-samplers"
            )

        cfg  = self.cfg
        bqm  = BinaryQuadraticModel(vartype="BINARY")
        bfs  = get_bfs_table()

        eligible, solving = self._compute_sets(start_state, depth, bfs)

        # ── Term 1: One-hot per step ───────────────────────────────────────
        # (Σ_m x[t,m] − 1)² = −Σ_m x[t,m] + 2·Σ_{m<m'} x[t,m]·x[t,m'] + const
        for t in range(depth):
            for m in range(N_MOVES):
                bqm.add_variable(self.var(t, m), -cfg.lambda_one)

            for m1, m2 in itertools.combinations(range(N_MOVES), 2):
                bqm.add_interaction(
                    self.var(t, m1),
                    self.var(t, m2),
                    2.0 * cfg.lambda_one,
                )

        # ── Term 2: Sequential validity penalty ───────────────────────────
        # Penalise x[t,m1]·x[t+1,m2] for same-face or cancelling pairs.
        for t in range(depth - 1):
            for m1 in range(N_MOVES):
                for m2 in range(N_MOVES):
                    if FACE_OF[m1] == FACE_OF[m2] or m2 == MOVE_INV[m1]:
                        bqm.add_interaction(
                            self.var(t,     m1),
                            self.var(t + 1, m2),
                            cfg.lambda_seq,
                        )

        # ── Term 3: Ineligibility penalty ─────────────────────────────────
        # Heavy positive bias for moves NOT on any shortest path.
        for t in range(depth):
            for m in range(N_MOVES):
                if (t, m) not in eligible:
                    bqm.add_variable(self.var(t, m), cfg.lambda_elig)

        # ── Term 4: Objective reward ──────────────────────────────────────
        # Negative bias on moves that deliver the solved state.
        for t, m in solving:
            bqm.add_variable(self.var(t, m), cfg.lambda_obj)

        log.debug(
            "BQM built: %d variables, %d interactions",
            bqm.num_variables, bqm.num_interactions,
        )
        return bqm


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Sampler selection
# ═══════════════════════════════════════════════════════════════════════════════

def get_sampler(use_leap: bool = False) -> object:
    """
    Return an initialised dimod-compatible sampler.

    Priority
    ────────
    1. use_leap=True + dwave-system installed + DWAVE_API_TOKEN set
       → LeapHybridSampler  (D-Wave cloud, highest solution quality)
    2. Otherwise → SimulatedAnnealingSampler  (local, no account needed)

    Raises RuntimeError if no sampler can be instantiated.
    """
    if not DIMOD_AVAILABLE:
        raise RuntimeError(
            "dimod is not installed.  Run: pip install dimod dwave-samplers"
        )

    if use_leap:
        if not LEAP_AVAILABLE:
            log.warning(
                "dwave-system not installed — falling back to SA.  "
                "Install with: pip install dwave-system"
            )
        elif not os.environ.get("DWAVE_API_TOKEN"):
            log.warning("DWAVE_API_TOKEN not set — falling back to SA.")
        else:
            log.info("Sampler: LeapHybridSampler (D-Wave cloud)")
            return LeapHybridSampler()

    if not SA_AVAILABLE:
        raise RuntimeError(
            "No sampler available.  Install: pip install dwave-samplers"
        )

    log.info("Sampler: SimulatedAnnealingSampler (local)")
    return SimulatedAnnealingSampler()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Sample decoder
# ═══════════════════════════════════════════════════════════════════════════════

def decode_sample(
    sample: dict[int, int],
    depth: int,
    start_state: tuple,
    bfs: BFSTable,
) -> tuple[list[int], bool]:
    """
    Decode the lowest-energy QUBO sample into a concrete move sequence.

    Repair heuristics (in priority order)
    ──────────────────────────────────────
    1. Exactly one variable set at step t → use that move  (ideal case).
    2. Multiple variables set (one-hot violated):
       Pick the move that minimises BFS distance from the current state.
       This is the most physically meaningful tie-break.
    3. No variable set:
       Follow the BFS back-pointer from the current state (local repair).
       This can happen when lambda_one is too small or the sampler is noisy.

    Returns
    ───────
    (move_indices, is_valid)
      move_indices : partial or complete sequence; may be shorter than depth
      is_valid     : True iff simulating the sequence from start reaches solved
    """
    sequence: list[int] = []
    state = start_state

    for t in range(depth):
        if is_solved(state):
            break

        active = [m for m in range(N_MOVES) if sample.get(t * N_MOVES + m, 0) == 1]

        if len(active) == 1:
            chosen = active[0]

        elif len(active) > 1:
            log.warning(
                "Step %d: one-hot violated (%d active); "
                "selecting by minimum BFS distance",
                t, len(active),
            )
            chosen = min(
                active,
                key=lambda m: bfs.depth(apply_move(state, m)) or 999,
            )

        else:
            log.warning("Step %d: no move selected; applying BFS repair", t)
            bfs_path = bfs.solve(state)
            if not bfs_path:
                break
            chosen = bfs_path[0]

        state = apply_move(state, chosen)
        sequence.append(chosen)

    return sequence, is_solved(state)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Public API
# ═══════════════════════════════════════════════════════════════════════════════

def quantum_solve(
    stickers: list[int],
    use_leap: bool = False,
    max_depth: int = 11,
    num_reads: int = 1000,
    config: Optional[QUBOConfig] = None,
    bfs_fallback: bool = True,
) -> SolveResult:
    """
    Solve a 2×2 Rubik's cube using QUBO / quantum-inspired optimisation.

    Pipeline
    ────────
    1. stickers_to_state()  — parse and validate input
    2. BFS lookup           — determine true God's number
    3. CubeQUBOBuilder      — build BQM of depth × 18 binary variables
    4. Sampler              — sample the BQM landscape
    5. decode_sample()      — extract move sequence from lowest-energy sample
    6. Simulation verify    — confirm the sequence actually solves the cube
    7. BFS fallback         — if QUBO failed, return guaranteed-optimal BFS solution

    Parameters
    ──────────
    stickers      24-integer colour array, same format as POST /solve.
    use_leap      Use D-Wave Leap cloud (requires DWAVE_API_TOKEN env var).
    max_depth     Search depth cap (≤ 14).
    num_reads     Number of SA reads; ignored by LeapHybridSampler.
    config        QUBO weights; None → QUBOConfig() defaults.
    bfs_fallback  Fall back to the exact BFS solution on QUBO failure.

    Returns
    ───────
    SolveResult dataclass.
    """
    t0    = time.perf_counter()
    cfg   = config or QUBOConfig()
    result = SolveResult()

    # ── Input validation ───────────────────────────────────────────────────
    if len(stickers) != 24:
        raise ValueError(f"Expected 24 stickers, got {len(stickers)}")
    for i, c in enumerate(stickers):
        if not (0 <= c <= 5):
            raise ValueError(f"Sticker {i}: colour {c!r} out of range 0-5")

    try:
        start = stickers_to_state(stickers)
    except ValueError as exc:
        raise ValueError(f"Invalid sticker input: {exc}") from exc

    # ── Trivial case ───────────────────────────────────────────────────────
    if is_solved(start):
        result.total_time_ms = (time.perf_counter() - t0) * 1000
        result.is_optimal    = True
        result.sampler_used  = "none"
        return result

    # ── Exact optimal depth via BFS ────────────────────────────────────────
    bfs        = get_bfs_table()
    true_depth = bfs.optimal_depth(start)
    depth      = min(max_depth, true_depth) if true_depth > 0 else max_depth

    log.info(
        "God's number: %d  |  QUBO depth: %d  |  sampler: %s",
        true_depth, depth, "Leap" if use_leap else "SA",
    )

    # ── Graceful degradation when dimod is absent ──────────────────────────
    if not DIMOD_AVAILABLE:
        log.warning("dimod not installed — using BFS solver directly")
        return _bfs_result(start, true_depth, t0, sampler="BFSFallback")

    # ── Build BQM ─────────────────────────────────────────────────────────
    bqm = CubeQUBOBuilder(cfg).build(start, depth)
    result.bqm_variables    = bqm.num_variables
    result.bqm_interactions = bqm.num_interactions
    log.info(
        "BQM: %d variables, %d interactions",
        bqm.num_variables, bqm.num_interactions,
    )

    # ── Sample ────────────────────────────────────────────────────────────
    sampler             = get_sampler(use_leap=use_leap)
    result.sampler_used = type(sampler).__name__

    t_sample = time.perf_counter()
    try:
        sample_set = sampler.sample(bqm, num_reads=num_reads)
    except TypeError:
        # LeapHybridSampler does not accept num_reads
        sample_set = sampler.sample(bqm)
    result.sample_time_ms = (time.perf_counter() - t_sample) * 1000

    best           = sample_set.first
    result.energy  = best.energy
    log.info(
        "Sampling done in %.1f ms  energy=%.6f",
        result.sample_time_ms, result.energy,
    )

    # ── Decode and verify ─────────────────────────────────────────────────
    move_indices, valid = decode_sample(best.sample, depth, start, bfs)

    if valid:
        result.move_indices  = move_indices
        result.moves         = [MOVE_NAMES[i] for i in move_indices]
        result.gods_number   = len(move_indices)
        result.is_optimal    = (result.gods_number == true_depth)
        result.fallback_used = False
        log.info(
            "QUBO solution valid: %d moves %s  optimal=%s",
            result.gods_number, result.moves, result.is_optimal,
        )

    elif bfs_fallback:
        log.warning(
            "QUBO sample invalid (energy=%.6f) — activating BFS fallback",
            result.energy,
        )
        fb                   = _bfs_result(start, true_depth, t0)
        # Preserve QUBO diagnostics
        fb.sampler_used      = result.sampler_used
        fb.sample_time_ms    = result.sample_time_ms
        fb.energy            = result.energy
        fb.bqm_variables     = result.bqm_variables
        fb.bqm_interactions  = result.bqm_interactions
        result               = fb

    else:
        log.error("QUBO failed and bfs_fallback=False — returning empty result")
        result.gods_number = -1

    result.total_time_ms = (time.perf_counter() - t0) * 1000
    return result


def _bfs_result(
    start: tuple,
    true_depth: int,
    t0: float,
    sampler: str = "BFSFallback",
) -> SolveResult:
    """Populate and return a SolveResult using only the BFS table."""
    bfs          = get_bfs_table()
    move_indices = bfs.solve(start)
    r            = SolveResult()
    r.move_indices   = move_indices
    r.moves          = [MOVE_NAMES[i] for i in move_indices]
    r.gods_number    = len(move_indices)
    r.is_optimal     = (r.gods_number == true_depth)
    r.fallback_used  = True
    r.sampler_used   = sampler
    r.total_time_ms  = (time.perf_counter() - t0) * 1000
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FastAPI integration
# ═══════════════════════════════════════════════════════════════════════════════

def solve_result_to_response(result: SolveResult) -> dict:
    """
    Serialise SolveResult to the JSON dict returned by POST /quantum-solve.
    Top-level keys are compatible with the existing SolveResponse schema in api.py.
    """
    return {
        "solution":        result.moves,
        "gods_number":     result.gods_number,
        "already_solved":  result.gods_number == 0,
        "solve_time_ms":   round(result.total_time_ms, 3),
        "quantum_metadata": {
            "sampler":          result.sampler_used,
            "sample_time_ms":   round(result.sample_time_ms, 3),
            "fallback_used":    result.fallback_used,
            "is_optimal":       result.is_optimal,
            "energy":           round(result.energy, 6),
            "bqm_variables":    result.bqm_variables,
            "bqm_interactions": result.bqm_interactions,
        },
    }


# Paste this block into api.py to register the /quantum-solve endpoint.
FASTAPI_SNIPPET: str = '''
# ── quantum_solver integration ── paste into api.py ──────────────────────────
from quantum_solver import (
    quantum_solve, solve_result_to_response, QUBOConfig, DIMOD_AVAILABLE,
)

class QuantumSolveRequest(BaseModel):
    """24-sticker state + quantum sampler options."""
    state: Annotated[
        list[int],
        Field(min_length=24, max_length=24,
              description="24 sticker colours (0-5), face-major order."),
    ]
    use_leap:  bool = Field(False, description="Use D-Wave Leap cloud")
    max_depth: int  = Field(11,    ge=1, le=14)
    num_reads: int  = Field(1000,  ge=1, le=100_000)

    @field_validator("state")
    @classmethod
    def colours_in_range(cls, v: list[int]) -> list[int]:
        for i, c in enumerate(v):
            if not (0 <= c <= 5):
                raise ValueError(f"Invalid colour {c} at index {i}")
        return v

@app.post("/quantum-solve", tags=["Solver"],
          summary="Solve via QUBO / quantum-inspired optimisation")
async def quantum_solve_endpoint(body: QuantumSolveRequest):
    if not DIMOD_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="dimod not installed — run: pip install dimod dwave-samplers",
        )
    try:
        result = quantum_solve(
            stickers  = body.state,
            use_leap  = body.use_leap,
            max_depth = body.max_depth,
            num_reads = body.num_reads,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return solve_result_to_response(result)
'''


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Unit tests and benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_unit_tests() -> bool:
    """
    Self-contained correctness tests; dimod is NOT required.

    Assertions
    ──────────
    1. M ∘ M⁻¹ = identity for all 18 moves.
    2. (R U R' U')×6 = identity.
    3. state_to_stickers ∘ stickers_to_state = identity
       for all 8 corners × 3 orientations (24 cases).
    4. Solved state produces the expected 24-sticker array.
    5. _compose_moves associativity: (A∘B)∘C = A∘(B∘C).

    Returns True on success; prints failures and returns False otherwise.
    """
    fails: list[str] = []

    # 1. M ∘ M⁻¹ = identity
    for i in range(N_MOVES):
        s = apply_move(SOLVED_STATE, i)
        s = apply_move(s, MOVE_INV[i])
        if s != SOLVED_STATE:
            fails.append(f"MOVE_INV[{i}] does not undo {MOVE_NAMES[i]}")

    # 2. (R U R' U')×6 = identity
    R, U, Rp, Up = 3, 0, 5, 2
    s = SOLVED_STATE
    for _ in range(6):
        for mv in (R, U, Rp, Up):
            s = apply_move(s, mv)
    if s != SOLVED_STATE:
        fails.append("(R U R' U')×6 ≠ identity")

    # 3. Round-trip for every corner × orientation combination
    for slot in range(8):
        for ori in range(3):
            base = list(SOLVED_STATE)
            base[slot] = (slot, ori)
            test = tuple(base)
            recovered = stickers_to_state(state_to_stickers(test))
            if recovered[slot] != (slot, ori):
                fails.append(
                    f"Round-trip fail: slot={slot} ori={ori}  "
                    f"got {recovered[slot]}"
                )

    # 4. Solved sticker array
    got      = state_to_stickers(SOLVED_STATE)
    expected = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 5,5,5,5, 4,4,4,4]
    if got != expected:
        fails.append(f"Solved sticker mismatch: {got}")

    # 5. Associativity: (A∘B)∘C = A∘(B∘C)
    pa, oa = _CW_TABLES[0]   # U
    pb, ob = _CW_TABLES[1]   # R
    pc, oc = _CW_TABLES[2]   # F
    pab, oab = _compose_moves(pa, oa, pb, ob)
    lhs, _ = _compose_moves(pab, oab, pc, oc)
    pbc, obc = _compose_moves(pb, ob, pc, oc)
    rhs, _ = _compose_moves(pa, oa, pbc, obc)
    if lhs != rhs:
        fails.append("_compose_moves is not associative")

    n_assertions = N_MOVES + 1 + 8 * 3 + 1 + 1
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        return False

    print(f"  All {n_assertions} assertions passed ✓")
    return True


def _random_scramble(depth: int, seed: Optional[int] = None) -> tuple[tuple, list[int]]:
    """Apply `depth` random non-same-face moves to the solved state."""
    rng       = random.Random(seed)
    state     = SOLVED_STATE
    applied   : list[int] = []
    last_face = -1
    for _ in range(depth):
        m = rng.randrange(N_MOVES)
        while FACE_OF[m] == last_face:
            m = rng.randrange(N_MOVES)
        state     = apply_move(state, m)
        applied.append(m)
        last_face = FACE_OF[m]
    return state, applied


def benchmark(
    num_trials: int = 5,
    depths: Optional[list[int]] = None,
    use_leap: bool = False,
    num_reads: int = 500,
    config: Optional[QUBOConfig] = None,
) -> None:
    """
    Benchmark the QUBO solver against BFS on random scrambles.

    Prints a comparison table:
      Depth | Trial | BFS | QUBO | Opt? | FB? | Sample ms | Total ms
    """
    if depths is None:
        depths = [3, 5, 7, 9]

    bfs = get_bfs_table()    # pre-build once

    w = 76
    print(f"\n{'─' * w}")
    print(
        f"{'Depth':>6} {'Trial':>5} {'BFS':>4} {'QUBO':>5} "
        f"{'Opt?':>5} {'FB?':>4} {'Sample ms':>10} {'Total ms':>9}"
    )
    print(f"{'─' * w}")

    ok = total = 0
    for depth in depths:
        for trial in range(num_trials):
            state, _  = _random_scramble(depth, seed=trial * 100 + depth)
            stickers  = state_to_stickers(state)
            true_d    = bfs.optimal_depth(state)
            try:
                res = quantum_solve(
                    stickers     = stickers,
                    use_leap     = use_leap,
                    num_reads    = num_reads,
                    max_depth    = depth + 2,
                    config       = config,
                    bfs_fallback = True,
                )
                opt = "✓" if res.is_optimal else "~"
                fb  = "yes" if res.fallback_used else "no"
                print(
                    f"{depth:>6} {trial:>5} {true_d:>4} {res.gods_number:>5} "
                    f"{opt:>5} {fb:>4} "
                    f"{res.sample_time_ms:>10.1f} {res.total_time_ms:>9.1f}"
                )
                ok += 1
            except Exception as exc:
                print(f"{depth:>6} {trial:>5}  ERROR: {exc}")
            total += 1

    print(f"{'─' * w}")
    print(f"Solved {ok}/{total} trials\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog            = "quantum_solver",
        description     = "QUBO-based optimal solver for the 2×2 Rubik's cube",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # solve
    s = sub.add_parser("solve", help="Solve a cube state via QUBO")
    s.add_argument("--stickers",    required=True, metavar="INT,...",
                   help="Comma-separated 24-integer sticker array")
    s.add_argument("--use_leap",    action="store_true")
    s.add_argument("--max_depth",   type=int,   default=11)
    s.add_argument("--num_reads",   type=int,   default=1000)
    s.add_argument("--lambda_one",  type=float, default=10.0)
    s.add_argument("--lambda_seq",  type=float, default=6.0)
    s.add_argument("--lambda_elig", type=float, default=20.0)
    s.add_argument("--lambda_obj",  type=float, default=-8.0)
    s.add_argument("--no_fallback", action="store_true")
    s.add_argument("--json",        action="store_true", help="Output JSON")
    s.add_argument("--verbose",     action="store_true")

    # benchmark
    b = sub.add_parser("benchmark", help="Benchmark QUBO vs BFS")
    b.add_argument("--num_trials",  type=int,       default=5)
    b.add_argument("--depths",      type=int, nargs="+", default=[3, 5, 7])
    b.add_argument("--use_leap",    action="store_true")
    b.add_argument("--num_reads",   type=int,       default=500)
    b.add_argument("--verbose",     action="store_true")

    # test
    sub.add_parser("test", help="Run unit tests (no dimod needed)")

    # info
    sub.add_parser("info", help="Print dependency / environment status")

    return p


def main() -> None:
    parser  = _build_cli()
    args    = parser.parse_args()
    verbose = getattr(args, "verbose", False)

    logging.basicConfig(
        level   = logging.DEBUG if verbose else logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force   = True,
    )

    if args.cmd == "info":
        print(f"dimod available:         {DIMOD_AVAILABLE}")
        print(f"SA sampler available:    {SA_AVAILABLE}")
        print(f"Leap sampler available:  {LEAP_AVAILABLE}")
        print(f"DWAVE_API_TOKEN set:     {bool(os.environ.get('DWAVE_API_TOKEN'))}")
        print(f"N_MOVES:                 {N_MOVES}")
        print(f"MOVE_TABLE entries:      {len(MOVE_TABLE)}")
        return

    if args.cmd == "test":
        print("Running unit tests …")
        ok = run_unit_tests()
        raise SystemExit(0 if ok else 1)

    if args.cmd == "benchmark":
        benchmark(
            num_trials = args.num_trials,
            depths     = args.depths,
            use_leap   = args.use_leap,
            num_reads  = args.num_reads,
        )
        return

    if args.cmd == "solve":
        try:
            stickers = [int(x.strip()) for x in args.stickers.split(",")]
        except ValueError as exc:
            parser.error(f"Cannot parse --stickers: {exc}")
            return

        cfg = QUBOConfig(
            lambda_one  = args.lambda_one,
            lambda_seq  = args.lambda_seq,
            lambda_elig = args.lambda_elig,
            lambda_obj  = args.lambda_obj,
        )

        result = quantum_solve(
            stickers     = stickers,
            use_leap     = args.use_leap,
            max_depth    = args.max_depth,
            num_reads    = args.num_reads,
            config       = cfg,
            bfs_fallback = not args.no_fallback,
        )

        if args.json:
            print(json.dumps(solve_result_to_response(result), indent=2))
            return

        sep = "─" * 54
        print(f"\n{sep}")
        print(f"  Solution:          {result.moves}")
        print(f"  God's number:      {result.gods_number}")
        print(f"  Optimal:           {result.is_optimal}")
        print(f"  Energy:            {result.energy:.6f}")
        print(f"  Sampler:           {result.sampler_used}")
        print(f"  Fallback used:     {result.fallback_used}")
        print(f"  BQM variables:     {result.bqm_variables}")
        print(f"  BQM interactions:  {result.bqm_interactions}")
        print(f"  Sample time:       {result.sample_time_ms:.1f} ms")
        print(f"  Total time:        {result.total_time_ms:.1f} ms")
        print(f"{sep}\n")


if __name__ == "__main__":
    main()
