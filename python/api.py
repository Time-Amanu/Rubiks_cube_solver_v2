"""
api.py — 2×2 Cube Solver FastAPI Coordinator
=============================================

Exposes the Rust cube2_solver extension over HTTP so any frontend
(including the Three.js widget) can POST a 24-sticker state and
receive the optimal move sequence back as JSON.

Run
---
    # Development (auto-reload)
    uvicorn api:app --reload --port 8000

    # Production
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

Endpoints
---------
    POST /solve           — optimal BFS solve (Rust engine)
    POST /gods-number     — minimum move count only
    POST /quantum-solve   — QUBO solve via D-Wave / SimulatedAnnealing
    GET  /health          — liveness check + component status
    GET  /docs            — Swagger UI (auto-generated)
    GET  /redoc           — ReDoc UI (auto-generated)
"""

from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager
from collections import Counter
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Rust extension ────────────────────────────────────────────────────────────
# Built with:  maturin develop --release   (inside rust_solver/)
# The module lives at cube2_solver.so / cube2_solver.pyd after compilation.
try:
    import cube2_solver as _solver
    SOLVER_AVAILABLE = True
except ImportError:
    _solver = None  # type: ignore[assignment]
    SOLVER_AVAILABLE = False

# ── Quantum solver (D-Wave QUBO) ──────────────────────────────────────────────
# Optional: requires  pip install dimod dwave-samplers
# Set DWAVE_API_TOKEN env var to use the Leap cloud sampler instead of SA.
try:
    from quantum_solver import (
        quantum_solve,
        solve_result_to_response,
        QUBOConfig,
        DIMOD_AVAILABLE,
    )
    QUANTUM_AVAILABLE = True
except ImportError:
    quantum_solve = None          # type: ignore[assignment]
    solve_result_to_response = None  # type: ignore[assignment]
    QUBOConfig = None             # type: ignore[assignment]
    DIMOD_AVAILABLE = False
    QUANTUM_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("cube2_api")

# ── Color & move metadata ─────────────────────────────────────────────────────
COLOR_NAMES: dict[int, str] = {
    0: "White",
    1: "Yellow",
    2: "Green",
    3: "Blue",
    4: "Red",
    5: "Orange",
}

FACE_LABELS: list[str] = ["Up", "Down", "Front", "Back", "Left", "Right"]

# Sticker layout for the unfolded diagram (used in error messages / health)
#   Indices  0– 3 = Up    face
#   Indices  4– 7 = Down  face
#   Indices  8–11 = Front face
#   Indices 12–15 = Back  face
#   Indices 16–19 = Left  face
#   Indices 20–23 = Right face

# ── Lifespan: warm up BFS table once at startup ───────────────────────────────

_table_ready = False
_table_build_ms: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the BFS table before the server accepts requests."""
    global _table_ready, _table_build_ms
    if SOLVER_AVAILABLE:
        log.info("Building BFS lookup table (all 3,674,160 states)…")
        t0 = time.perf_counter()
        _solver.warmup()
        _table_build_ms = round((time.perf_counter() - t0) * 1000, 1)
        _table_ready = True
        log.info("BFS table ready in %.1f ms — solver online.", _table_build_ms)
    else:
        log.warning(
            "cube2_solver Rust extension NOT found. "
            "Build it with: cd cube2_solver && maturin develop --release"
        )
    yield
    log.info("Shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="2×2 Rubik's Cube Solver API",
    description=(
        "Optimal solver for the 2×2 pocket cube. "
        "Uses a pre-computed BFS table (Rust/PyO3) to return God's-number "
        "solutions for any valid scramble in under 1 ms."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allows the Three.js frontend (any localhost port, file://, or your domain)
# to call the API directly from the browser.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",   # Create React App / Vite default
        "http://localhost:5173",   # Vite
        "http://localhost:8080",   # common dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        # Add your production domain here, e.g.:
        # "https://your-app.example.com",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,  # cache preflight for 10 min
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    """
    24-element sticker state of the cube.

    Color encoding
    --------------
    0 = White   (Up face when solved)
    1 = Yellow  (Down face when solved)
    2 = Green   (Front face when solved)
    3 = Blue    (Back face when solved)
    4 = Red     (Right face when solved)
    5 = Orange  (Left face when solved)

    Array layout
    ------------
    [ 0.. 3] Up    — back-left, back-right, front-left, front-right
    [ 4.. 7] Down  — front-left, front-right, back-left, back-right
    [ 8..11] Front — top-left, top-right, bottom-left, bottom-right
    [12..15] Back  — top-left, top-right, bottom-left, bottom-right
    [16..19] Left  — top-back, top-front, bottom-back, bottom-front
    [20..23] Right — top-front, top-back, bottom-front, bottom-back
    """

    state: Annotated[
        list[int],
        Field(
            min_length=24,
            max_length=24,
            description="24 sticker color integers (0–5), face-major order.",
            examples=[[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 5,5,5,5, 4,4,4,4]],
        ),
    ]

    @field_validator("state")
    @classmethod
    def colors_in_range(cls, v: list[int]) -> list[int]:
        for i, c in enumerate(v):
            if not (0 <= c <= 5):
                raise ValueError(
                    f"Invalid color {c!r} at index {i}. "
                    f"Must be 0–5 ({', '.join(f'{k}={n}' for k,n in COLOR_NAMES.items())})."
                )
        return v

    @model_validator(mode="after")
    def each_color_appears_four_times(self) -> "SolveRequest":
        counts = Counter(self.state)
        bad = {
            f"{COLOR_NAMES[c]} ({c})": cnt
            for c, cnt in counts.items()
            if cnt != 4
        }
        if bad:
            detail = "; ".join(f"{name} appears {cnt}× (expected 4)" for name, cnt in bad.items())
            raise ValueError(f"Invalid sticker distribution — {detail}.")
        return self


class SolveResponse(BaseModel):
    """Successful solve result."""

    solution: list[str] = Field(
        description="Optimal move sequence, e.g. ['R', \"U'\", 'F2'].",
        examples=[["R", "U'", "F2", "L"]],
    )
    gods_number: int = Field(
        ge=0, le=14,
        description="Minimum moves required (God's number for this state).",
        examples=[4],
    )
    already_solved: bool = Field(
        description="True when the submitted state was already solved.",
    )
    solve_time_ms: float = Field(
        description="Wall-clock time spent inside the Rust solver (milliseconds).",
        examples=[0.042],
    )


class GodsNumberResponse(BaseModel):
    """God's number only — cheaper than returning the full solution."""

    gods_number: int = Field(ge=0, le=14)
    solve_time_ms: float


class HealthResponse(BaseModel):
    status: str
    solver_available: bool
    bfs_table_ready: bool
    bfs_table_build_ms: float | None
    quantum_available: bool


class QuantumSolveRequest(BaseModel):
    """24-sticker state + quantum sampler options."""
    state: Annotated[
        list[int],
        Field(
            min_length=24,
            max_length=24,
            description="24 sticker colors (0-5), face-major order.",
            examples=[[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 5,5,5,5, 4,4,4,4]],
        ),
    ]
    use_leap:  bool = Field(False,  description="Use D-Wave Leap cloud sampler (needs DWAVE_API_TOKEN)")
    max_depth: int  = Field(11,     ge=1, le=14, description="Maximum search depth")
    num_reads: int  = Field(1000,   ge=1, le=100_000, description="SA reads (ignored by Leap)")

    @field_validator("state")
    @classmethod
    def colors_in_range(cls, v: list[int]) -> list[int]:
        for i, c in enumerate(v):
            if not (0 <= c <= 5):
                raise ValueError(f"Invalid color {c!r} at index {i}. Must be 0-5.")
        return v

    @model_validator(mode="after")
    def each_color_appears_four_times(self) -> "QuantumSolveRequest":
        counts = Counter(self.state)
        bad = {
            f"{COLOR_NAMES[c]} ({c})": cnt
            for c, cnt in counts.items()
            if cnt != 4
        }
        if bad:
            detail = "; ".join(f"{name} appears {cnt}× (expected 4)" for name, cnt in bad.items())
            raise ValueError(f"Invalid sticker distribution — {detail}.")
        return self


class ErrorDetail(BaseModel):
    error: str
    detail: str | None = None


# ── Helper ────────────────────────────────────────────────────────────────────

def _require_solver() -> None:
    """Raise 503 if the Rust extension isn't loaded."""
    if not SOLVER_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Solver unavailable",
                "detail": (
                    "The cube2_solver Rust extension is not installed. "
                    "Build it with: cd cube2_solver && maturin develop --release"
                ),
            },
        )
    if not _table_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "BFS table not ready",
                "detail": "The server is still initialising. Retry in a moment.",
            },
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness & readiness check",
    tags=["Meta"],
)
async def health() -> HealthResponse:
    """
    Returns the operational status of the API and whether the BFS table
    has been successfully built.
    """
    return HealthResponse(
        status="ok" if (_table_ready and SOLVER_AVAILABLE) else "degraded",
        solver_available=SOLVER_AVAILABLE,
        bfs_table_ready=_table_ready,
        bfs_table_build_ms=_table_build_ms if _table_ready else None,
        quantum_available=QUANTUM_AVAILABLE,
    )


@app.post(
    "/solve",
    response_model=SolveResponse,
    summary="Solve a 2×2 cube state",
    tags=["Solver"],
    responses={
        200: {"description": "Optimal solution found."},
        400: {"model": ErrorDetail, "description": "Invalid sticker input."},
        422: {"description": "Request body failed validation."},
        503: {"model": ErrorDetail, "description": "Solver not available."},
    },
)
async def solve(body: SolveRequest) -> SolveResponse:
    """
    Accepts a 24-element sticker array and returns the **optimal** move
    sequence using the pre-computed BFS lookup table built from the complete
    2×2 state space (~3.67 M states).

    **Performance:** table build ≈ 50 ms once at startup; each query ≈ 0.01–0.1 ms.

    **Move notation** (standard WCA / SiGN):

    | Symbol | Meaning                    |
    |--------|----------------------------|
    | `U`    | Up face clockwise          |
    | `U'`   | Up face counter-clockwise  |
    | `U2`   | Up face 180°               |
    | `R`    | Right face clockwise       |
    | …      | (same pattern for F D L B) |
    """
    _require_solver()

    stickers = body.state

    try:
        t0 = time.perf_counter()
        solution: list[str] = _solver.solve(stickers)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    except ValueError as exc:
        # Rust raised PyValueError — invalid or unreachable state
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Solver rejected the input", "detail": str(exc)},
        ) from exc
    except Exception as exc:
        log.exception("Unexpected solver error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal solver error", "detail": str(exc)},
        ) from exc

    log.info(
        "Solved in %d moves (%.3f ms) | state=%s",
        len(solution), elapsed_ms, stickers[:4],  # log first face only
    )

    return SolveResponse(
        solution=solution,
        gods_number=len(solution),
        already_solved=(len(solution) == 0),
        solve_time_ms=elapsed_ms,
    )


@app.post(
    "/gods-number",
    response_model=GodsNumberResponse,
    summary="Return God's number without the full solution",
    tags=["Solver"],
    responses={
        400: {"model": ErrorDetail, "description": "Invalid sticker input."},
        503: {"model": ErrorDetail, "description": "Solver not available."},
    },
)
async def gods_number_endpoint(body: SolveRequest) -> GodsNumberResponse:
    """
    Cheaper than `/solve` when you only need the minimum move count and not
    the actual move sequence.  Same validation rules apply.
    """
    _require_solver()

    try:
        t0 = time.perf_counter()
        gn: int = _solver.gods_number(body.state)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Solver rejected the input", "detail": str(exc)},
        ) from exc

    return GodsNumberResponse(gods_number=gn, solve_time_ms=elapsed_ms)


# ── Quantum solve endpoint ────────────────────────────────────────────────────

@app.post(
    "/quantum-solve",
    summary="Solve via QUBO / quantum-inspired optimisation",
    tags=["Solver"],
    responses={
        200: {"description": "Solution found (may use BFS fallback if QUBO fails)."},
        400: {"model": ErrorDetail, "description": "Invalid sticker input."},
        503: {"model": ErrorDetail, "description": "Quantum solver not available."},
    },
)
async def quantum_solve_endpoint(body: QuantumSolveRequest) -> dict:
    """
    Builds a Binary Quadratic Model from the cube state and samples it with
    SimulatedAnnealingSampler (local, no account needed) or LeapHybridSampler
    (D-Wave cloud — set `use_leap=true` and `DWAVE_API_TOKEN` env var).

    Always returns a valid solution: if the QUBO sample does not produce a
    legal path it falls back to the BFS-optimal Rust solver automatically.

    Install quantum deps:  `pip install dimod dwave-samplers`
    """
    if not QUANTUM_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Quantum solver unavailable",
                "detail": (
                    "dimod is not installed. "
                    "Run: pip install dimod dwave-samplers"
                ),
            },
        )

    try:
        result = quantum_solve(
            stickers     = body.state,
            use_leap     = body.use_leap,
            max_depth    = body.max_depth,
            num_reads    = body.num_reads,
            bfs_fallback = True,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Solver rejected the input", "detail": str(exc)},
        ) from exc
    except Exception as exc:
        log.exception("Quantum solver error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal quantum solver error", "detail": str(exc)},
        ) from exc

    return solve_result_to_response(result)


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
