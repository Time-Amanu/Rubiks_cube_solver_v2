<div align="center">

<!-- Animated 2×2 Rubik's Cube (renders directly on GitHub, no image hosting needed) -->
<img src="https://raw.githubusercontent.com/Time-Amanu/Rubiks_cube_solver_v2/main/assets/cube.svg" width="180" alt="Animated 2×2 Rubik's Cube"/>
<h1>2×2 Rubik's Cube Solver</h1>

<p>
  An end-to-end system that solves any 2×2 Rubik's cube state optimally — from reading sticker colors in a photo, to computing the shortest possible move sequence, to returning it over a REST API in under 1 millisecond.
</p>

<!-- Badges -->
<p>
  <img src="https://img.shields.io/badge/Rust-1.77%2B-orange?style=flat-square&logo=rust" alt="Rust"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/D--Wave-Ocean%20SDK-6E2FBF?style=flat-square" alt="D-Wave"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Tests-74%20passing-brightgreen?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/Solve%20time-%3C1ms-brightgreen?style=flat-square" alt="Speed"/>
</p>

</div>

---

## What this does

Take a scrambled 2×2 Rubik's cube. Take a photo. This system:

1. **Classifies each sticker color** from the photo using a fine-tuned MobileNetV3 neural network
2. **Computes the optimal solution** — the absolute minimum number of moves possible — using a pre-computed BFS table over all 3,674,160 reachable cube states, implemented in Rust
3. **Returns the move sequence** via a REST API in under 1 millisecond
4. **Optionally solves it on quantum hardware** by framing the problem as a Binary Quadratic Model (QUBO) for D-Wave's Leap platform

---

## Architecture

```
📷 Camera / 3D Widget
       │
       │  readCubeState.js
       │  (converts 3D state → 24-int array)
       ▼
┌─────────────────────────────────────────────┐
│           FastAPI  (api.py)                 │
│  • Pydantic validation                      │
│  • CORS for localhost frontends             │
│  • POST /solve  /quantum-solve  /health     │
└────────────┬─────────────────┬──────────────┘
             │                 │
     ┌───────▼──────┐   ┌──────▼────────────────┐
     │  Rust solver  │   │  Quantum solver        │
     │  (lib.rs)     │   │  (quantum_solver.py)   │
     │               │   │                        │
     │  BFS table:   │   │  Binary Quadratic      │
     │  3.67M states │   │  Model → D-Wave Leap   │
     │  ~50ms build  │   │  or SimulatedAnnealing │
     │  <1ms query   │   │                        │
     └───────────────┘   └────────────────────────┘

🧠 ML Pipeline (rubiks_color_classifier.py)
   MobileNetV3-Small fine-tuned → 6-class sticker classifier
   infer_face(photo) → [0, 4, 2, 1]  (4 sticker colors)
```

---

## Project structure

```
rubiks-cube-solver/
├── rust_solver/
│   ├── src/lib.rs                 ← BFS solver + PyO3 bindings (Rust)
│   ├── Cargo.toml
│   └── pyproject.toml             ← maturin build config
├── python/
│   ├── api.py                     ← FastAPI: all endpoints
│   ├── quantum_solver.py          ← D-Wave QUBO solver
│   ├── cube2_solver.py            ← Python wrapper
│   └── requirements.txt
├── ml/
│   └── rubiks_color_classifier.py ← PyTorch training + inference
├── frontend/
│   └── readCubeState.js           ← Three.js → sticker array bridge
├── tests/
│   ├── test_api.py                ← 26 API tests
│   └── test_quantum_solver.py     ← 74 solver tests
└── assets/
    └── cube.svg                   ← Animated cube (this header)
```

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/solve` | Optimal BFS solve — returns move sequence + God's number |
| `POST` | `/gods-number` | Minimum move count only (faster) |
| `POST` | `/quantum-solve` | QUBO solve via D-Wave SA or Leap cloud |
| `GET`  | `/health` | Component status — Rust solver, BFS table, quantum |
| `GET`  | `/docs` | Live Swagger UI — test all endpoints in browser |

### Example request

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"state": [4,2,4,5,3,1,0,3,4,0,3,2,1,1,5,2,0,5,3,1,2,4,0,5]}'
```

```json
{
  "solution": ["R", "U'", "F2", "L", "D", "R'", "U"],
  "gods_number": 7,
  "already_solved": false,
  "solve_time_ms": 0.031
}
```

---

## Color encoding

The 24-element sticker array uses this mapping — consistent across all components:

| Integer | Color | Face when solved |
|---------|-------|-----------------|
| `0` | ⬜ White | Up |
| `1` | 🟨 Yellow | Down |
| `2` | 🟩 Green | Front |
| `3` | 🟦 Blue | Back |
| `4` | 🟥 Red | Right |
| `5` | 🟧 Orange | Left |

---

## How it works

### Rust BFS solver

The 2×2 cube has exactly **3,674,160** reachable states. At startup, a Breadth-First Search runs from the solved state, expanding all 18 moves and storing `(depth, back_move)` for every state discovered. This builds a complete lookup table in ~50 ms. Any solve query is then just a table lookup + pointer traversal — no search at query time.

Each state is encoded as a single `u64` — 7 corners × 5 bits (3 bits for slot position + 2 bits for orientation). The 8th corner is derived, since all orientations must sum to 0 mod 3.

### Quantum QUBO solver

The solve problem is encoded as a Binary Quadratic Model:

```
Variables:  x[t, m] ∈ {0,1}  — "apply move m at step t?"
             t ∈ {0…D-1},  m ∈ {0…17}  (18 HTM moves)

Energy:  E = λ₁·P_one  +  λ₂·P_seq  +  λ₃·P_elig  +  λ₄·P_obj
```

- **P_one** — one-hot constraint: exactly one move per step
- **P_seq** — penalise same-face and cancelling consecutive moves
- **P_elig** — heavy penalty for moves not on any shortest path
- **P_obj** — reward for the move sequence that reaches solved

The sampler finds the lowest-energy configuration → the optimal move sequence.

### ML color classifier

MobileNetV3-Small pre-trained on ImageNet, fine-tuned in two phases:
1. **Head-only** (3 epochs, lr=1e-3) — backbone frozen, only the new 6-class head trains
2. **Full fine-tuning** (remaining epochs, lr=1e-4) — entire network adapts to cube stickers

`WeightedRandomSampler` handles class imbalance. `infer_face()` splits a face photo into a 2×2 grid and classifies each patch independently.

---

## Getting started

### Prerequisites

- Python 3.10+
- Rust (stable) — [rustup.rs](https://rustup.rs)

### 1. Install Python packages

```bash
pip install fastapi uvicorn pydantic maturin dimod dwave-samplers
```

### 2. Build the Rust extension

```bash
cd rust_solver
maturin develop --release
cd ..
```

Takes 1–3 minutes. When you see `Installed cube2_solver-0.1.0` it's done.

### 3. Verify

```bash
python -c "import cube2_solver; print('Rust solver ready!')"
```

### 4. Start the API

```bash
cd python
uvicorn api:app --reload --port 8000
```

### 5. Open in browser

```
http://localhost:8000/docs
```

You'll see the live Swagger UI. Click any endpoint → Try it out → Execute.

---

## Running the tests

```bash
# Quantum solver unit tests (no dimod required)
python python/quantum_solver.py test

# Full test suite (requires pytest + httpx)
pip install pytest httpx
pytest tests/ -v
```

Expected: **74 quantum solver assertions** + **26 API tests** all passing.

---

## ML classifier (optional)

Train on your own Kaggle sticker dataset:

```bash
# Expects data/train/ and data/val/ with subfolders per color
python ml/rubiks_color_classifier.py train \
    --data_dir ./data --epochs 15 --model mobilenet

# Classify a face photo
python ml/rubiks_color_classifier.py infer \
    --image face.jpg --checkpoint best_model.pt

# Debug misclassified stickers
python ml/rubiks_color_classifier.py debug \
    --data_dir ./data --checkpoint best_model.pt
```

---

## Quantum solver (optional)

Requires `pip install dimod dwave-samplers`. For D-Wave Leap cloud, set `DWAVE_API_TOKEN`.

```bash
# Local simulated annealing (no account needed)
python python/quantum_solver.py solve \
    --stickers 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,5,5,5,5,4,4,4,4

# D-Wave Leap cloud
python python/quantum_solver.py solve --stickers ... --use_leap

# Benchmark QUBO vs BFS
python python/quantum_solver.py benchmark --depths 3 5 7 9
```

---

## Key numbers

| Metric | Value |
|--------|-------|
| Reachable cube states | 3,674,160 |
| BFS table build time | ~50 ms (once at startup) |
| Solve query time | < 1 ms |
| QUBO variables (max) | 198 (11 steps × 18 moves) |
| Test coverage | 100 tests across 2 suites |
| God's number (2×2 HTM) | 11 moves |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Optimal solver | Rust + PyO3 (via maturin) |
| Web API | FastAPI + Pydantic + Uvicorn |
| Quantum optimizer | D-Wave Ocean SDK (dimod + dwave-samplers) |
| ML classifier | PyTorch + MobileNetV3-Small + torchvision |
| 3D frontend bridge | Vanilla JavaScript (Three.js compatible) |
| Testing | pytest + FastAPI TestClient + httpx |

---

## Move notation

Standard WCA / SiGN notation:

| Move | Meaning |
|------|---------|
| `U` | Up face clockwise |
| `U'` | Up face counter-clockwise |
| `U2` | Up face 180° |
| `R F D L B` | Right, Front, Down, Left, Back — same variants |

18 moves total. Maximum solution length: 11 moves (God's number).

---

<div align="center">
<sub>Built with Rust · Python · D-Wave Ocean · PyTorch</sub>
</div>