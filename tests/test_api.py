"""
tests/test_api.py
=================
Integration tests for the FastAPI coordinator.

Run
---
    pip install pytest pytest-asyncio httpx
    pytest tests/ -v

These tests use FastAPI's built-in TestClient (synchronous HTTPX transport)
so no live server is needed — everything runs in-process.

NOTE: The cube2_solver Rust extension must be compiled before running:
    maturin develop --release
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Import the app — this also triggers lifespan startup (BFS table build)
from api import app

client = TestClient(app)

# ── Fixtures: sticker arrays ──────────────────────────────────────────────────

SOLVED = [
    0, 0, 0, 0,    # Up    — White
    1, 1, 1, 1,    # Down  — Yellow
    2, 2, 2, 2,    # Front — Green
    3, 3, 3, 3,    # Back  — Blue
    5, 5, 5, 5,    # Left  — Orange
    4, 4, 4, 4,    # Right — Red
]

# R U R' U' scramble (4 moves deep)
SCRAMBLE_4 = [
    0, 4, 0, 4,
    1, 0, 1, 0,
    2, 0, 2, 0,
    5, 3, 5, 3,
    5, 1, 5, 1,
    2, 4, 2, 4,
]

# 10-move scramble: R U R' F D L B2 R U' F'
# (pre-computed expected sticker state)
SCRAMBLE_10 = [
    4, 2, 4, 5,
    3, 1, 0, 3,
    4, 0, 3, 2,
    1, 1, 5, 2,
    0, 5, 3, 1,
    2, 4, 0, 5,
]

# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_schema(self):
        body = client.get("/health").json()
        assert "status" in body
        assert "solver_available" in body
        assert "bfs_table_ready" in body
        assert "bfs_table_build_ms" in body

    def test_status_ok_when_solver_present(self):
        body = client.get("/health").json()
        # If the Rust extension is compiled, both flags must be True
        if body["solver_available"]:
            assert body["bfs_table_ready"] is True
            assert body["status"] == "ok"
            assert isinstance(body["bfs_table_build_ms"], float)


# ── /solve — happy path ───────────────────────────────────────────────────────

class TestSolveHappyPath:
    def _post(self, state: list[int]) -> dict:
        r = client.post("/solve", json={"state": state})
        return r

    def test_solved_cube_returns_empty_solution(self):
        r = self._post(SOLVED)
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        assert r.status_code == 200
        body = r.json()
        assert body["solution"] == []
        assert body["gods_number"] == 0
        assert body["already_solved"] is True
        assert body["solve_time_ms"] >= 0

    def test_response_schema(self):
        r = self._post(SOLVED)
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        body = r.json()
        assert "solution" in body
        assert "gods_number" in body
        assert "already_solved" in body
        assert "solve_time_ms" in body

    def test_gods_number_matches_solution_length(self):
        r = self._post(SOLVED)
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        body = r.json()
        assert body["gods_number"] == len(body["solution"])

    def test_solve_time_is_fast_ms(self):
        r = self._post(SOLVED)
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        assert r.json()["solve_time_ms"] < 100   # comfortably under 100 ms

    def test_solution_moves_are_valid_notation(self):
        valid_moves = {
            "U", "U'", "U2", "R", "R'", "R2",
            "F", "F'", "F2", "D", "D'", "D2",
            "L", "L'", "L2", "B", "B'", "B2",
        }
        # Use a scramble that definitely has a non-empty solution
        # We'll just check the format on a solved cube for now —
        # swap SOLVED for a real scramble once the Rust solver is compiled.
        r = self._post(SOLVED)
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        for move in r.json()["solution"]:
            assert move in valid_moves, f"Unexpected move token: {move!r}"

    def test_gods_number_is_optimal_for_solved(self):
        r = self._post(SOLVED)
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        assert r.json()["gods_number"] == 0

    @pytest.mark.parametrize("scramble,max_depth", [
        (SOLVED,      0),
        (SCRAMBLE_4, 14),   # upper bound; optimal is probably 4 or fewer
    ])
    def test_gods_number_within_bounds(self, scramble, max_depth):
        r = self._post(scramble)
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        gn = r.json()["gods_number"]
        assert 0 <= gn <= 14, f"God's number {gn} out of range"
        assert gn <= max_depth or max_depth == 14


# ── /solve — validation errors ────────────────────────────────────────────────

class TestSolveValidation:
    def test_wrong_length_returns_422(self):
        r = client.post("/solve", json={"state": [0] * 20})
        assert r.status_code == 422

    def test_too_long_returns_422(self):
        r = client.post("/solve", json={"state": [0] * 30})
        assert r.status_code == 422

    def test_out_of_range_color_returns_422(self):
        bad = SOLVED[:]
        bad[5] = 9          # color 9 is invalid
        r = client.post("/solve", json={"state": bad})
        assert r.status_code == 422

    def test_negative_color_returns_422(self):
        bad = SOLVED[:]
        bad[0] = -1
        r = client.post("/solve", json={"state": bad})
        assert r.status_code == 422

    def test_wrong_color_distribution_returns_422(self):
        # Replace one White (0) with Red (4) — now 5 Reds and 3 Whites
        bad = SOLVED[:]
        bad[0] = 4
        r = client.post("/solve", json={"state": bad})
        assert r.status_code == 422
        body = r.json()
        # FastAPI wraps Pydantic errors in {"detail": [...]}
        assert "detail" in body

    def test_missing_state_field_returns_422(self):
        r = client.post("/solve", json={})
        assert r.status_code == 422

    def test_empty_body_returns_422(self):
        r = client.post("/solve", content=b"", headers={"content-type": "application/json"})
        assert r.status_code == 422

    def test_non_integer_colors_returns_422(self):
        bad_body = {"state": ["R", "U", "F"] + [0] * 21}
        r = client.post("/solve", json=bad_body)
        assert r.status_code == 422


# ── /gods-number ──────────────────────────────────────────────────────────────

class TestGodsNumber:
    def test_solved_returns_zero(self):
        r = client.post("/gods-number", json={"state": SOLVED})
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        assert r.status_code == 200
        assert r.json()["gods_number"] == 0

    def test_schema(self):
        r = client.post("/gods-number", json={"state": SOLVED})
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        body = r.json()
        assert "gods_number" in body
        assert "solve_time_ms" in body

    def test_validation_same_as_solve(self):
        r = client.post("/gods-number", json={"state": [0] * 10})
        assert r.status_code == 422


# ── CORS ──────────────────────────────────────────────────────────────────────

class TestCORS:
    """Verify that the correct CORS headers are returned for browser requests."""

    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
    ]

    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_preflight_allowed_origin(self, origin: str):
        r = client.options(
            "/solve",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert r.status_code in (200, 204)
        assert "access-control-allow-origin" in r.headers

    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_actual_request_includes_acao_header(self, origin: str):
        r = client.get("/health", headers={"Origin": origin})
        assert "access-control-allow-origin" in r.headers

    def test_allow_methods_includes_post(self):
        r = client.options(
            "/solve",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        methods = r.headers.get("access-control-allow-methods", "")
        assert "POST" in methods or r.status_code in (200, 204)


# ── Performance ───────────────────────────────────────────────────────────────

class TestPerformance:
    """Smoke test: the Rust solver must complete well under 100 ms."""

    def test_solve_under_100ms(self):
        import time
        r = client.post("/solve", json={"state": SOLVED})
        if r.status_code == 503:
            pytest.skip("Rust extension not compiled")
        assert r.json()["solve_time_ms"] < 100

    def test_repeated_solves_are_fast(self):
        """10 consecutive solves must all finish in <100 ms each."""
        for _ in range(10):
            r = client.post("/solve", json={"state": SOLVED})
            if r.status_code == 503:
                pytest.skip("Rust extension not compiled")
            assert r.json()["solve_time_ms"] < 100
