/**
 * readCubeState()
 * ===============
 * Reads the current orientation of the 2×2 pocket cube and returns a flat
 * array of 24 integers representing sticker colors in a fixed face order.
 *
 * COLOR ENCODING
 * ──────────────
 *   0 = White
 *   1 = Yellow
 *   2 = Green
 *   3 = Blue
 *   4 = Red
 *   5 = Orange
 *
 * FACE ORDER & INDEX MAP
 * ──────────────────────
 * The array is partitioned into 6 consecutive blocks of 4 stickers each:
 *
 *   Indices  0– 3  →  Up    (U) face
 *   Indices  4– 7  →  Down  (D) face
 *   Indices  8–11  →  Front (F) face
 *   Indices 12–15  →  Back  (B) face
 *   Indices 16–19  →  Left  (L) face
 *   Indices 20–23  →  Right (R) face
 *
 * STICKER ORDER WITHIN EACH FACE
 * ────────────────────────────────
 * Each face block lists its 4 stickers in reading order (row-major, left→right,
 * top→bottom) as seen by a viewer standing directly in front of that face:
 *
 *   +---+---+
 *   | 0 | 1 |    e.g. for Up face:  index 0 = back-left  sticker
 *   +---+---+                        index 1 = back-right sticker
 *   | 2 | 3 |                        index 2 = front-left sticker
 *   +---+---+                        index 3 = front-right sticker
 *
 * Applied to all faces (viewer faces the face head-on, cube in standard
 * orientation: White on top, Green in front):
 *
 *   Face   Base  [+0]          [+1]          [+2]          [+3]
 *   ─────  ────  ────────────  ────────────  ────────────  ─────────────
 *   Up      0    back-left     back-right    front-left    front-right
 *   Down    4    front-left    front-right   back-left     back-right
 *   Front   8    top-left      top-right     bottom-left   bottom-right
 *   Back   12    top-left      top-right     bottom-left   bottom-right
 *   Left   16    top-back      top-front     bottom-back   bottom-front
 *   Right  20    top-front     top-back      bottom-front  bottom-back
 *
 * Note: "top/bottom" on side faces refers to proximity to the U/D face.
 *       "front/back" on U/D refers to proximity to the F/B face.
 *       "left/right" is always from the viewer's perspective facing that face.
 *
 * INTERNAL STATE → OUTPUT MAPPING
 * ────────────────────────────────
 * This widget stores state in `cubeState` (Uint8Array, 24 elements) using the
 * internal engine's color encoding:
 *   engine: 0=White, 1=Red, 2=Green, 3=Yellow, 4=Orange, 5=Blue
 *
 * The internal face order is: U(0-3), R(4-7), F(8-11), D(12-15), L(16-19), B(20-23)
 *
 * This function remaps both the face order and color values to the external spec.
 *
 * INTERNAL → EXTERNAL COLOR MAP:
 *   engine 0 (White)  → output 0
 *   engine 1 (Red)    → output 4
 *   engine 2 (Green)  → output 2
 *   engine 3 (Yellow) → output 1
 *   engine 4 (Orange) → output 5
 *   engine 5 (Blue)   → output 3
 *
 * INTERNAL → EXTERNAL STICKER INDEX REMAP:
 * The output array index maps to this internal cubeState index:
 *
 *   Out[ 0] = cubeState[ 0]   (U face, back-left)
 *   Out[ 1] = cubeState[ 1]   (U face, back-right)
 *   Out[ 2] = cubeState[ 2]   (U face, front-left)
 *   Out[ 3] = cubeState[ 3]   (U face, front-right)
 *   Out[ 4] = cubeState[12]   (D face, front-left)
 *   Out[ 5] = cubeState[13]   (D face, front-right)
 *   Out[ 6] = cubeState[14]   (D face, back-left)
 *   Out[ 7] = cubeState[15]   (D face, back-right)
 *   Out[ 8] = cubeState[ 8]   (F face, top-left)
 *   Out[ 9] = cubeState[ 9]   (F face, top-right)
 *   Out[10] = cubeState[10]   (F face, bottom-left)
 *   Out[11] = cubeState[11]   (F face, bottom-right)
 *   Out[12] = cubeState[20]   (B face, top-left)
 *   Out[13] = cubeState[21]   (B face, top-right)
 *   Out[14] = cubeState[22]   (B face, bottom-left)
 *   Out[15] = cubeState[23]   (B face, bottom-right)
 *   Out[16] = cubeState[16]   (L face, top-back)
 *   Out[17] = cubeState[17]   (L face, top-front)
 *   Out[18] = cubeState[18]   (L face, bottom-back)
 *   Out[19] = cubeState[19]   (L face, bottom-front)
 *   Out[20] = cubeState[ 4]   (R face, top-front)
 *   Out[21] = cubeState[ 5]   (R face, top-back)
 *   Out[22] = cubeState[ 6]   (R face, bottom-front)
 *   Out[23] = cubeState[ 7]   (R face, bottom-back)
 *
 * @param {Uint8Array|number[]} cubeState - The internal 24-element engine state.
 * @returns {number[]} Flat array of 24 integers in [0..5], external spec format.
 *
 * @example
 * // Solved cube should return:
 * // U(0-3)=White(0), D(4-7)=Yellow(1), F(8-11)=Green(2),
 * // B(12-15)=Blue(3), L(16-19)=Orange(5), R(20-23)=Red(4)
 * readCubeState(SOLVED);
 * // → [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 5,5,5,5, 4,4,4,4]
 */
function readCubeState(cubeState) {
  // Internal engine color → external output color
  const COLOR_REMAP = [
    0, // engine 0 (White)  → 0 (White)
    4, // engine 1 (Red)    → 4 (Red)
    2, // engine 2 (Green)  → 2 (Green)
    1, // engine 3 (Yellow) → 1 (Yellow)
    5, // engine 4 (Orange) → 5 (Orange)
    3, // engine 5 (Blue)   → 3 (Blue)
  ];

  // Maps output index [0..23] → internal cubeState index
  // Face blocks: U=0-3, D=4-7, F=8-11, B=12-15, L=16-19, R=20-23
  const INDEX_REMAP = [
    // Up face (internal indices 0-3, already match)
     0,  1,  2,  3,
    // Down face (internal indices 12-15)
    12, 13, 14, 15,
    // Front face (internal indices 8-11, already match)
     8,  9, 10, 11,
    // Back face (internal indices 20-23)
    20, 21, 22, 23,
    // Left face (internal indices 16-19, already match)
    16, 17, 18, 19,
    // Right face (internal indices 4-7)
     4,  5,  6,  7,
  ];

  return INDEX_REMAP.map(internalIdx => COLOR_REMAP[cubeState[internalIdx]]);
}


// ─── BACKEND PARSING REFERENCE ───────────────────────────────────────────────
//
// Your backend receives an array of 24 integers. Parse it like this:
//
//   const state = readCubeState(cubeState);
//
//   const UP    = state.slice( 0,  4);  // [back-left, back-right, front-left, front-right]
//   const DOWN  = state.slice( 4,  8);  // [front-left, front-right, back-left, back-right]
//   const FRONT = state.slice( 8, 12);  // [top-left, top-right, bottom-left, bottom-right]
//   const BACK  = state.slice(12, 16);  // [top-left, top-right, bottom-left, bottom-right]
//   const LEFT  = state.slice(16, 20);  // [top-back, top-front, bottom-back, bottom-front]
//   const RIGHT = state.slice(20, 24);  // [top-front, top-back, bottom-front, bottom-back]
//
// Solved state assertion:
//   UP.every(c => c === 0)    // all White
//   DOWN.every(c => c === 1)  // all Yellow
//   FRONT.every(c => c === 2) // all Green
//   BACK.every(c => c === 3)  // all Blue
//   LEFT.every(c => c === 5)  // all Orange
//   RIGHT.every(c => c === 4) // all Red

// ── Module export (works in both Node.js and browser bundlers) ────────────────
// Browser (plain script tag): readCubeState is already a global function.
// Node.js / CommonJS:
if (typeof module !== "undefined" && module.exports) {
  module.exports = { readCubeState, COLOR_REMAP, INDEX_REMAP };
}
// ES Module (import { readCubeState } from "./readCubeState.js"):
// Add  type="module"  to your <script> tag, then use the export below.
// export { readCubeState, COLOR_REMAP, INDEX_REMAP };

