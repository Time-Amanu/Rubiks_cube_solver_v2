"""
cube2_solver.py
===============
Python wrapper around the cube2_solver Rust extension module.

Usage
-----
    from cube2_solver import Cube2Solver

    solver = Cube2Solver()          # builds BFS table once (~50 ms)

    stickers = [
        0,0,0,0,   # Up    (White)
        1,1,1,1,   # Down  (Yellow)
        2,2,2,2,   # Front (Green)
        3,3,3,3,   # Back  (Blue)
        5,5,5,5,   # Left  (Orange)
        4,4,4,4,   # Right (Red)
    ]
    solution = solver.solve(stickers)   # e.g. ["R", "U'", "F2"]
    depth    = solver.gods_number(stickers)

Color encoding
--------------
    0 = White   (Up   face when solved)
    1 = Yellow  (Down face when solved)
    2 = Green   (Front face when solved)
    3 = Blue    (Back  face when solved)
    4 = Red     (Right face when solved)
    5 = Orange  (Left  face when solved)

Sticker array layout (24 elements, face-major)
-----------------------------------------------
    [ 0.. 3] = Up    face: back-left, back-right, front-left, front-right
    [ 4.. 7] = Down  face: front-left, front-right, back-left, back-right
    [ 8..11] = Front face: top-left, top-right, bottom-left, bottom-right
    [12..15] = Back  face: top-left, top-right, bottom-left, bottom-right
    [16..19] = Left  face: top-back, top-front, bottom-back, bottom-front
    [20..23] = Right face: top-front, top-back, bottom-front, bottom-back

    ("top/bottom" = closer to U/D face; "front/back" = closer to F/B face;
     all positions from the perspective of a viewer facing that face.)
"""

from __future__ import annotations
import time
from typing import List

# The Rust extension module — built with: maturin develop --release
try:
    import cube2_solver as _ext
except ImportError as e:
    raise ImportError(
        "Rust extension not found. Build it with:\n"
        "  pip install maturin\n"
        "  maturin develop --release\n"
        f"Original error: {e}"
    ) from e


SOLVED_STICKERS: List[int] = [
    0, 0, 0, 0,   # Up    – White
    1, 1, 1, 1,   # Down  – Yellow
    2, 2, 2, 2,   # Front – Green
    3, 3, 3, 3,   # Back  – Blue
    5, 5, 5, 5,   # Left  – Orange
    4, 4, 4, 4,   # Right – Red
]

COLOR_NAMES = {0: "White", 1: "Yellow", 2: "Green",
               3: "Blue",  4: "Red",    5: "Orange"}

FACE_NAMES  = ["Up", "Down", "Front", "Back", "Left", "Right"]


class Cube2Solver:
    """
    Optimal 2×2 Rubik's Cube solver backed by a full BFS pre-computation.

    The BFS table is built once on first instantiation (or on first call to
    `solve`/`gods_number`).  Subsequent calls are O(solution_depth) — just
    pointer-chasing through the table.

    Performance targets:
        Table build : ~50 ms  (covers all 3,674,160 reachable states)
        Any query   : <1 ms   (including 14-move worst case)
    """

    def __init__(self, warmup: bool = True) -> None:
        """
        Parameters
        ----------
        warmup : bool
            If True (default), trigger BFS table construction immediately.
            Set False to defer until the first solve() call.
        """
        self._built = False
        if warmup:
            self._ensure_ready()

    def _ensure_ready(self) -> None:
        if not self._built:
            t0 = time.perf_counter()
            _ext.warmup()
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[cube2_solver] BFS table ready ({elapsed:.1f} ms)")
            self._built = True

    # ── Core API ──────────────────────────────────────────────────────────────

    def solve(self, stickers: List[int]) -> List[str]:
        """
        Return the optimal solution for the given cube state.

        Parameters
        ----------
        stickers : list of 24 ints in 0..5
            The cube's 24 sticker colors in the face order described above.

        Returns
        -------
        list of str
            Optimal sequence of moves, e.g. ["R", "U'", "F2"].
            Empty list if the cube is already solved.

        Raises
        ------
        ValueError
            If the input is malformed or the state is unreachable.
        """
        self._validate(stickers)
        self._ensure_ready()
        return _ext.solve(list(stickers))

    def gods_number(self, stickers: List[int]) -> int:
        """
        Return the minimum number of moves required to solve this state.

        Parameters
        ----------
        stickers : list of 24 ints in 0..5

        Returns
        -------
        int
            God's number for this specific state (0 = already solved, max 14).
        """
        self._validate(stickers)
        self._ensure_ready()
        return _ext.gods_number(list(stickers))

    def solve_timed(self, stickers: List[int]) -> dict:
        """
        Solve and return timing information.

        Returns
        -------
        dict with keys:
            solution    : list[str]  — move sequence
            gods_number : int        — length of solution
            solve_ms    : float      — time spent in Rust solver (ms)
        """
        self._validate(stickers)
        self._ensure_ready()
        t0 = time.perf_counter()
        solution = _ext.solve(list(stickers))
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "solution":    solution,
            "gods_number": len(solution),
            "solve_ms":    round(elapsed, 3),
        }

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(stickers: List[int]) -> None:
        if len(stickers) != 24:
            raise ValueError(f"Expected 24 stickers, got {len(stickers)}")
        for i, c in enumerate(stickers):
            if not (0 <= c <= 5):
                raise ValueError(
                    f"Invalid color {c} at index {i}: must be in 0..5"
                )
        # Check each color appears exactly 4 times
        from collections import Counter
        counts = Counter(stickers)
        for color in range(6):
            if counts[color] != 4:
                raise ValueError(
                    f"Color {color} ({COLOR_NAMES[color]}) appears "
                    f"{counts[color]} times; expected 4"
                )

    # ── Pretty printing ───────────────────────────────────────────────────────

    @staticmethod
    def pretty(stickers: List[int]) -> str:
        """Return a human-readable unfolded cube diagram."""
        cn = {0:'W',1:'Y',2:'G',3:'B',4:'R',5:'O'}
        s = [cn[c] for c in stickers]
        return (
            f"        {s[0]} {s[1]}\n"
            f"        {s[2]} {s[3]}   (Up)\n"
            f"\n"
            f"  {s[16]} {s[17]}  {s[8]} {s[9]}  {s[20]} {s[21]}  {s[12]} {s[13]}\n"
            f"  {s[18]} {s[19]}  {s[10]} {s[11]}  {s[22]} {s[23]}  {s[14]} {s[15]}\n"
            f"  (Left)  (Front) (Right)  (Back)\n"
            f"\n"
            f"        {s[4]} {s[5]}\n"
            f"        {s[6]} {s[7]}   (Down)\n"
        )


# ── Standalone test harness ───────────────────────────────────────────────────

def _run_tests():
    """Run built-in correctness and performance tests."""
    import time

    solver = Cube2Solver(warmup=True)

    print("\n─── Test suite ───────────────────────────────────────")

    tests = [
        ("Solved",           SOLVED_STICKERS[:]),
        ("R",                _scramble(["R"])),
        ("R U R' U'",        _scramble(["R","U","R'","U'"])),
        ("10-move scramble",
            _scramble(["R","U","R'","F","D","L","B2","R","U'","F'"])),
        ("14-move scramble",
            _scramble(["F","R","U","R'","U'","F'","R","U","R'","U'","R","U","R'","U'"])),
    ]

    all_pass = True
    for name, stickers in tests:
        t0 = time.perf_counter()
        result = solver.solve_timed(stickers)
        elapsed = (time.perf_counter() - t0) * 1000
        solution = result["solution"]
        gn       = result["gods_number"]

        # Verify by applying solution and checking solved
        final = _apply_moves(stickers, solution)
        ok = (final == SOLVED_STICKERS)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False

        print(f"  [{status}] {name:<25} → {gn:2d} moves  {elapsed:6.2f} ms  {solution}")

    print("──────────────────────────────────────────────────────")
    print("All tests passed!" if all_pass else "SOME TESTS FAILED.")
    return all_pass


def _scramble(moves: List[str]) -> List[int]:
    """Apply move sequence to a solved cube and return stickers."""
    return _apply_moves(SOLVED_STICKERS[:], moves)


def _apply_moves(stickers: List[int], moves: List[str]) -> List[int]:
    """
    Pure-Python move application for test verification.
    Delegates to quantum_solver's verified cube engine (no Rust needed).
    This is independent of the Rust extension so tests can run without building it.
    """
    try:
        from quantum_solver import (
            stickers_to_state, apply_move, state_to_stickers, MOVE_NAMES
        )
    except ImportError:
        raise ImportError(
            "quantum_solver.py must be in the same directory as cube2_solver.py. "
            "It provides the pure-Python cube engine used for test verification."
        )

    state = stickers_to_state(stickers)
    for mv_name in moves:
        if mv_name not in MOVE_NAMES:
            raise ValueError(f"Unknown move: {mv_name!r}. Valid moves: {MOVE_NAMES}")
        idx = MOVE_NAMES.index(mv_name)
        state = apply_move(state, idx)
    return state_to_stickers(state)


if __name__ == "__main__":
    _run_tests()
