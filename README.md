# Quantum Data Center Optimization

A graduation project demonstrating quantum-enhanced optimization for data center operations — real-time energy management, anomaly detection, and workload scheduling using Qiskit, PyTorch, and FastAPI.

---

## Features

| Module | Algorithm | Description |
|--------|-----------|-------------|
| Task Allocation | **QAOA** (Quantum Approximate Optimization Algorithm) | Assigns tasks to servers to minimize energy consumption |
| Workload Scheduling | **VQE** (Variational Quantum Eigensolver) | Schedules workloads during low-cost, low-carbon hours |
| Cooling Optimization | **QUBO** (Quadratic Unconstrained Binary Optimization) | Finds optimal temperature setpoints per zone |
| Energy Forecasting | **LSTM** (PyTorch, 2-layer) | Predicts next-24h energy consumption from workload + temperature |
| Anomaly Detection | **Classical SVM** vs **Quantum Kernel SVM** (ZZFeatureMap) | Detects abnormal sensor readings; compares both approaches |
| Carbon Optimizer | Classical scheduling heuristic | Shifts flexible workloads to green (low-carbon) hours |

---

## Quantum Advantage

The benchmark panel runs the same task-allocation problem with:
- **Classical Greedy** — deterministic, O(n log n)
- **QAOA** — explores all allocations simultaneously via quantum superposition

Results are displayed side-by-side with energy improvement %, execution time, and the actual QAOA circuit diagram.

---

## Setup

```bash
# 1. Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main dashboard |
| GET | `/report` | Thesis export page (printable) |
| GET | `/api/status` | Current simulation state + metrics |
| GET | `/api/servers` | Server rack status |
| GET | `/api/tasks` | Task queue with allocation |
| GET | `/api/timeseries` | 24h energy/carbon/temperature data |
| POST | `/api/optimize` | Run QAOA / VQE / QUBO / Greedy on demand |
| GET | `/api/benchmark` | QAOA vs Greedy side-by-side comparison |
| GET | `/api/forecast` | LSTM 24h energy forecast |
| GET | `/api/carbon` | Carbon schedule optimizer |
| GET | `/api/model-metrics` | LSTM R², RMSE + SVM comparison |
| GET | `/api/history` | Historical metrics log (last 200 steps) |
| GET | `/api/history/export` | Download full history as CSV |
| POST | `/api/simulation/start` | Start simulation |
| POST | `/api/simulation/stop` | Stop simulation |
| POST | `/api/simulation/reset` | Reset simulation + clear history |
| WS | `/ws` | Real-time metrics WebSocket |

---

## Project Structure

```
quantumCode/
├── app/
│   ├── main.py              # FastAPI backend — all endpoints
│   └── static/
│       ├── index.html       # Dashboard UI
│       ├── app.js           # Frontend logic
│       ├── style.css        # Dark theme styles
│       └── favicon.svg      # Quantum atom favicon
├── src/quantum_dc/
│   ├── optimization/
│   │   ├── task_allocation.py       # QAOA optimizer
│   │   ├── workload_scheduling.py   # VQE scheduler
│   │   └── cooling_optimization.py  # QUBO cooling
│   ├── prediction/
│   │   └── energy_predictor.py      # LSTM forecaster (PyTorch)
│   ├── learning/
│   │   └── anomaly_detector.py      # Classical + Quantum Kernel SVM
│   └── utils/
│       └── data_generator.py        # Synthetic data generation
└── requirements.txt
```

---

## Dashboard Panels

1. **Simulation Control** — Start/Stop/Reset + optimization mode selector
2. **Live Metrics** — Energy, cost, temperature, carbon, PUE, utilization (WebSocket, 1s updates)
3. **Server Rack** — Visual utilization per server (color-coded: green/yellow/red)
4. **Task Queue** — Active tasks with server assignments
5. **Energy & Cost Forecast** — 24h price chart
6. **Server Load Distribution** — Bar chart per server
7. **Quantum vs Classical Benchmark** — QAOA vs Greedy with circuit diagram
8. **Quantum Kernel SVM vs Classical SVM** — Anomaly detection accuracy comparison
9. **LSTM Energy Forecast** — Predicted vs actual energy chart
10. **Historical Metrics** — Multi-line chart of full session + CSV export
11. **Carbon Optimizer** — Hour-by-hour carbon schedule with savings
12. **Thesis Report** (`/report`) — Printable summary of all results

---

## Requirements

- Python 3.10+
- Qiskit 1.x + qiskit-algorithms + qiskit-optimization + qiskit-machine-learning
- PyTorch 2.x
- scikit-learn, numpy, scipy, pandas
- FastAPI + uvicorn

See `requirements.txt` for pinned versions.
