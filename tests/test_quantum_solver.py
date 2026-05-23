"""
tests/test_quantum_solver.py
============================
Pytest suite for quantum_solver.py.

Run
───
    pip install pytest
    pytest tests/test_quantum_solver.py -v

These tests do NOT require dimod, dwave-system, or a D-Wave account.
All QUBO-dependent tests are skipped automatically when dimod is absent.
"""

from __future__ import annotations

import random
import pytest

import quantum_solver as qs

# ── Fixtures ──────────────────────────────────────────────────────────────────

SOLVED_STICKERS = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 5,5,5,5, 4,4,4,4]

def _apply_moves(stickers: list[int], moves: list[str]) -> list[int]:
    """Apply a sequence of move-name strings to a sticker array."""
    state = qs.stickers_to_state(stickers)
    for mv in moves:
        idx = qs.MOVE_NAMES.index(mv)
        state = qs.apply_move(state, idx)
    return qs.state_to_stickers(state)

def _scramble_stickers(depth: int, seed: int = 0) -> list[int]:
    state, _ = qs._random_scramble(depth, seed=seed)
    return qs.state_to_stickers(state)

# ── Section 2: Cube engine ────────────────────────────────────────────────────

class TestMoveTable:
    def test_table_has_18_entries(self):
        assert len(qs.MOVE_TABLE) == 18

    def test_move_names_count(self):
        assert len(qs.MOVE_NAMES) == qs.N_MOVES == 18

    def test_move_inv_length(self):
        assert len(qs.MOVE_INV) == 18

    def test_face_of_length(self):
        assert len(qs.FACE_OF) == 18

    @pytest.mark.parametrize("i", range(18))
    def test_move_inverse_is_identity(self, i):
        s = qs.apply_move(qs.SOLVED_STATE, i)
        s = qs.apply_move(s, qs.MOVE_INV[i])
        assert s == qs.SOLVED_STATE, \
            f"{qs.MOVE_NAMES[qs.MOVE_INV[i]]} did not undo {qs.MOVE_NAMES[i]}"

    def test_r_u_rp_up_order6(self):
        """(R U R' U')×6 = identity — classic verification."""
        R, U, Rp, Up = 3, 0, 5, 2
        s = qs.SOLVED_STATE
        for _ in range(6):
            for mv in (R, U, Rp, Up):
                s = qs.apply_move(s, mv)
        assert s == qs.SOLVED_STATE

    def test_each_face_order4(self):
        """Every quarter-turn applied 4 times returns to identity."""
        for face in range(6):
            cw_idx = face * 3       # CW move index
            s = qs.SOLVED_STATE
            for _ in range(4):
                s = qs.apply_move(s, cw_idx)
            assert s == qs.SOLVED_STATE, \
                f"Face {face} CW ×4 ≠ identity"

    def test_180_applied_twice_is_identity(self):
        for face in range(6):
            idx180 = face * 3 + 1
            s = qs.apply_move(qs.SOLVED_STATE, idx180)
            s = qs.apply_move(s, idx180)
            assert s == qs.SOLVED_STATE, \
                f"{qs.MOVE_NAMES[idx180]} ×2 ≠ identity"

    def test_solved_state_is_solved(self):
        assert qs.is_solved(qs.SOLVED_STATE)

    def test_apply_any_move_not_solved(self):
        for i in range(18):
            assert not qs.is_solved(qs.apply_move(qs.SOLVED_STATE, i)), \
                f"Applying {qs.MOVE_NAMES[i]} to solved should not be solved"

    def test_compose_associativity(self):
        """(A∘B)∘C  ==  A∘(B∘C)."""
        pa, oa = qs._CW_TABLES[0]   # U
        pb, ob = qs._CW_TABLES[1]   # R
        pc, oc = qs._CW_TABLES[2]   # F
        pab, oab = qs._compose_moves(pa, oa, pb, ob)
        lhs, lho = qs._compose_moves(pab, oab, pc, oc)
        pbc, obc = qs._compose_moves(pb, ob, pc, oc)
        rhs, rho = qs._compose_moves(pa, oa, pbc, obc)
        assert lhs == rhs
        assert lho == rho

    def test_state_key_unique(self):
        """All 18 one-move states from solved have distinct keys."""
        keys = {qs.state_to_key(qs.apply_move(qs.SOLVED_STATE, i)) for i in range(18)}
        assert len(keys) == 18

    def test_state_key_solved(self):
        k = qs.state_to_key(qs.SOLVED_STATE)
        assert isinstance(k, int)
        assert k >= 0


# ── Section 3: Sticker conversion ─────────────────────────────────────────────

class TestStickerConversion:
    def test_solved_stickers_match_expected(self):
        got = qs.state_to_stickers(qs.SOLVED_STATE)
        assert got == SOLVED_STICKERS

    def test_stickers_to_state_solved(self):
        state = qs.stickers_to_state(SOLVED_STICKERS)
        assert qs.is_solved(state)

    @pytest.mark.parametrize("slot", range(8))
    @pytest.mark.parametrize("ori", range(3))
    def test_round_trip_all_corners(self, slot, ori):
        """state_to_stickers ∘ stickers_to_state = identity for every (slot, ori)."""
        base = list(qs.SOLVED_STATE)
        base[slot] = (slot, ori)
        test_state = tuple(base)
        recovered = qs.stickers_to_state(qs.state_to_stickers(test_state))
        assert recovered[slot] == (slot, ori), \
            f"Round-trip failed: slot={slot} ori={ori} → got {recovered[slot]}"

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Expected 24"):
            qs.stickers_to_state([0] * 20)

    def test_bad_color_combo_raises(self):
        bad = SOLVED_STICKERS[:]
        bad[0] = 4   # now UBL has Red on its UD face — impossible piece
        with pytest.raises(ValueError):
            qs.stickers_to_state(bad)

    def test_each_face_has_one_color_when_solved(self):
        s = SOLVED_STICKERS
        assert len(set(s[0:4]))  == 1
        assert len(set(s[4:8]))  == 1
        assert len(set(s[8:12])) == 1
        assert len(set(s[12:16]))== 1
        assert len(set(s[16:20]))== 1
        assert len(set(s[20:24]))== 1

    def test_after_U_move_upper_face_still_one_color(self):
        """After U, the U face stickers change positions but remain White."""
        after = _apply_moves(SOLVED_STICKERS, ["U"])
        assert all(c == 0 for c in after[0:4]), "U face should stay White after U turn"

    def test_scramble_round_trip(self):
        """Random scramble → stickers → state → stickers should be self-consistent."""
        rng = random.Random(42)
        state = qs.SOLVED_STATE
        for _ in range(8):
            state = qs.apply_move(state, rng.randrange(18))
        stickers  = qs.state_to_stickers(state)
        recovered = qs.stickers_to_state(stickers)
        assert recovered == state


# ── Section 4: BFS table ──────────────────────────────────────────────────────

class TestBFSTable:
    @pytest.fixture(scope="class")
    def bfs(self):
        t = qs.BFSTable()
        t.build()
        return t

    def test_solved_depth_zero(self, bfs):
        assert bfs.depth(qs.SOLVED_STATE) == 0

    def test_one_move_depth_one(self, bfs):
        for i in range(18):
            s = qs.apply_move(qs.SOLVED_STATE, i)
            assert bfs.depth(s) == 1, \
                f"{qs.MOVE_NAMES[i]} from solved should be depth 1"

    def test_solve_returns_empty_for_solved(self, bfs):
        assert bfs.solve(qs.SOLVED_STATE) == []

    def test_solve_one_move(self, bfs):
        for i in range(18):
            s = qs.apply_move(qs.SOLVED_STATE, i)
            sol = bfs.solve(s)
            assert len(sol) == 1
            # Applying the solution should reach solved
            result = qs.apply_move(s, sol[0])
            assert qs.is_solved(result)

    @pytest.mark.parametrize("depth,seed", [(4,1),(6,2),(8,3),(10,4)])
    def test_solve_multi_move(self, bfs, depth, seed):
        state, _ = qs._random_scramble(depth, seed=seed)
        sol = bfs.solve(state)
        assert 1 <= len(sol) <= bfs.depth(state)
        # Verify solution
        s = state
        for mv in sol:
            s = qs.apply_move(s, mv)
        assert qs.is_solved(s)

    def test_optimal_depth_solved(self, bfs):
        assert bfs.optimal_depth(qs.SOLVED_STATE) == 0

    def test_len_nonzero(self, bfs):
        assert len(bfs) > 0


# ── Section 5 & 8: QUBO builder and quantum_solve ────────────────────────────

DIMOD_SKIP = pytest.mark.skipif(
    not qs.DIMOD_AVAILABLE,
    reason="dimod not installed",
)
SA_SKIP = pytest.mark.skipif(
    not qs.SA_AVAILABLE,
    reason="SimulatedAnnealingSampler not available",
)


@DIMOD_SKIP
class TestQUBOBuilder:
    @pytest.fixture
    def bfs(self):
        t = qs.BFSTable()
        t.build()
        return t

    def test_build_returns_bqm(self, bfs):
        state = qs.apply_move(qs.SOLVED_STATE, 0)   # depth-1 state
        bqm = qs.CubeQUBOBuilder(qs.QUBOConfig()).build(state, 1)
        assert bqm is not None
        assert bqm.num_variables > 0

    def test_variable_count(self, bfs):
        depth = 3
        state, _ = qs._random_scramble(depth, seed=7)
        bqm = qs.CubeQUBOBuilder(qs.QUBOConfig()).build(state, depth)
        # Must have at most depth * N_MOVES variables (pruning may reduce this)
        assert bqm.num_variables <= depth * qs.N_MOVES

    def test_no_negative_interaction_outside_seq(self):
        """Interactions within the same time step must all be positive (one-hot)."""
        state = qs.apply_move(qs.SOLVED_STATE, 3)   # R
        bqm = qs.CubeQUBOBuilder(qs.QUBOConfig()).build(state, 1)
        for (u, v), bias in bqm.quadratic.items():
            t_u = u // qs.N_MOVES
            t_v = v // qs.N_MOVES
            if t_u == t_v:
                assert bias >= 0, \
                    f"Same-step interaction ({u},{v}) has negative bias {bias}"

    def test_solved_state_no_bqm_needed(self):
        """quantum_solve on solved state returns immediately without building BQM."""
        result = qs.quantum_solve(SOLVED_STICKERS)
        assert result.gods_number == 0
        assert result.moves == []
        assert result.bqm_variables == 0


@SA_SKIP
@DIMOD_SKIP
class TestQuantumSolve:
    """End-to-end tests using SimulatedAnnealingSampler (local, no account)."""

    def _verify(self, stickers: list[int], result: qs.SolveResult):
        """Simulate result.moves from stickers and assert solved."""
        state = qs.stickers_to_state(stickers)
        for mv_name in result.moves:
            idx = qs.MOVE_NAMES.index(mv_name)
            state = qs.apply_move(state, idx)
        assert qs.is_solved(state), \
            f"Moves {result.moves} did not solve the cube"

    def test_solved_input(self):
        result = qs.quantum_solve(SOLVED_STICKERS)
        assert result.gods_number == 0
        assert result.is_optimal

    def test_single_move_scramble(self):
        for mv in ["U", "R", "F", "D", "L", "B"]:
            scrambled = _apply_moves(SOLVED_STICKERS, [mv])
            result = qs.quantum_solve(scrambled, num_reads=500, bfs_fallback=True)
            assert result.gods_number >= 1
            self._verify(scrambled, result)

    def test_four_move_scramble(self):
        scrambled = _apply_moves(SOLVED_STICKERS, ["R", "U", "R'", "U'"])
        result = qs.quantum_solve(scrambled, num_reads=1000, bfs_fallback=True)
        assert result.gods_number >= 1
        self._verify(scrambled, result)

    def test_result_schema(self):
        scrambled = _scramble_stickers(3, seed=99)
        result = qs.quantum_solve(scrambled, num_reads=200, bfs_fallback=True)
        assert isinstance(result.moves, list)
        assert isinstance(result.move_indices, list)
        assert isinstance(result.gods_number, int)
        assert isinstance(result.is_optimal, bool)
        assert isinstance(result.energy, float)
        assert isinstance(result.sample_time_ms, float)
        assert isinstance(result.total_time_ms, float)
        assert isinstance(result.sampler_used, str)
        assert isinstance(result.fallback_used, bool)
        assert result.total_time_ms >= result.sample_time_ms

    def test_move_names_are_valid(self):
        scrambled = _scramble_stickers(4, seed=11)
        result = qs.quantum_solve(scrambled, num_reads=500, bfs_fallback=True)
        for mv in result.moves:
            assert mv in qs.MOVE_NAMES, f"Unknown move name: {mv!r}"

    def test_move_indices_match_names(self):
        scrambled = _scramble_stickers(3, seed=22)
        result = qs.quantum_solve(scrambled, num_reads=500, bfs_fallback=True)
        assert len(result.moves) == len(result.move_indices)
        for name, idx in zip(result.moves, result.move_indices):
            assert qs.MOVE_NAMES[idx] == name

    def test_gods_number_equals_moves_length(self):
        scrambled = _scramble_stickers(5, seed=33)
        result = qs.quantum_solve(scrambled, num_reads=300, bfs_fallback=True)
        assert result.gods_number == len(result.moves)

    def test_bfs_fallback_always_valid(self):
        """With bfs_fallback=True, we must ALWAYS get a valid solution."""
        for seed in range(5):
            scrambled = _scramble_stickers(7, seed=seed)
            result = qs.quantum_solve(
                scrambled, num_reads=50, bfs_fallback=True
            )
            assert result.gods_number >= 0
            self._verify(scrambled, result)

    def test_invalid_sticker_length_raises(self):
        with pytest.raises(ValueError, match="Expected 24"):
            qs.quantum_solve([0] * 20)

    def test_invalid_color_raises(self):
        bad = SOLVED_STICKERS[:]
        bad[0] = 9
        with pytest.raises(ValueError):
            qs.quantum_solve(bad)

    def test_sampler_name_in_result(self):
        scrambled = _scramble_stickers(2, seed=5)
        result = qs.quantum_solve(scrambled, num_reads=100, bfs_fallback=True)
        assert result.sampler_used != ""

    def test_no_fallback_may_return_invalid(self):
        """With very few reads and no fallback, QUBO may fail silently."""
        scrambled = _scramble_stickers(8, seed=42)
        result = qs.quantum_solve(
            scrambled, num_reads=1, bfs_fallback=False
        )
        # We just check it doesn't raise — result may be invalid
        assert isinstance(result.gods_number, int)


# ── Section 9: FastAPI response helper ───────────────────────────────────────

class TestSolveResultToResponse:
    def test_solved_response(self):
        r = qs.SolveResult(
            moves=[], move_indices=[], gods_number=0,
            is_optimal=True, energy=0.0,
            sample_time_ms=0.1, total_time_ms=0.5,
            sampler_used="none", fallback_used=False,
        )
        resp = qs.solve_result_to_response(r)
        assert resp["solution"] == []
        assert resp["gods_number"] == 0
        assert resp["already_solved"] is True
        assert "quantum_metadata" in resp
        meta = resp["quantum_metadata"]
        assert meta["sampler"] == "none"
        assert meta["fallback_used"] is False
        assert meta["is_optimal"] is True

    def test_unsolved_response(self):
        r = qs.SolveResult(
            moves=["R", "U'"], move_indices=[3, 2], gods_number=2,
            is_optimal=True, energy=-5.5,
            sample_time_ms=12.3, total_time_ms=50.1,
            sampler_used="SimulatedAnnealingSampler", fallback_used=False,
            bqm_variables=36, bqm_interactions=120,
        )
        resp = qs.solve_result_to_response(r)
        assert resp["solution"] == ["R", "U'"]
        assert resp["gods_number"] == 2
        assert resp["already_solved"] is False
        assert resp["solve_time_ms"] == round(50.1, 3)
        meta = resp["quantum_metadata"]
        assert meta["energy"] == round(-5.5, 6)
        assert meta["bqm_variables"] == 36
        assert meta["bqm_interactions"] == 120

    def test_response_keys_present(self):
        r = qs.SolveResult()
        resp = qs.solve_result_to_response(r)
        top_keys = {"solution", "gods_number", "already_solved",
                    "solve_time_ms", "quantum_metadata"}
        assert top_keys.issubset(resp.keys())
        meta_keys = {"sampler", "sample_time_ms", "fallback_used",
                     "is_optimal", "energy", "bqm_variables", "bqm_interactions"}
        assert meta_keys.issubset(resp["quantum_metadata"].keys())


# ── Section 10: Unit test runner ──────────────────────────────────────────────

class TestBuiltinUnitTests:
    def test_run_unit_tests_passes(self):
        """The module's own self-test suite must pass cleanly."""
        assert qs.run_unit_tests() is True


# ── QUBOConfig dataclass ──────────────────────────────────────────────────────

class TestQUBOConfig:
    def test_defaults(self):
        cfg = qs.QUBOConfig()
        assert cfg.lambda_one  > 0
        assert cfg.lambda_seq  > 0
        assert cfg.lambda_elig > cfg.lambda_one   # must dominate one-hot
        assert cfg.lambda_obj  < 0                # must be a reward

    def test_custom_values(self):
        cfg = qs.QUBOConfig(lambda_one=20.0, lambda_seq=8.0,
                            lambda_elig=30.0, lambda_obj=-12.0)
        assert cfg.lambda_one  == 20.0
        assert cfg.lambda_seq  ==  8.0
        assert cfg.lambda_elig == 30.0
        assert cfg.lambda_obj  == -12.0
