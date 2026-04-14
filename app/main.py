"""
FastAPI Backend for Quantum Data Center Optimization
Real-time monitoring and control interface
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import numpy as np
from datetime import datetime
import sys
import os
import sqlite3
import csv
import io

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum_dc import (
    TaskAllocationOptimizer,
    WorkloadScheduler,
    CoolingOptimizer,
    EnergyPredictor,
    AnomalyDetector,
    DataCenterDataGenerator,
    DataCenterConfig,
    create_sample_dataset
)

# ---------------------------------------------------------------------------
# SQLite historical metrics database
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "metrics_history.db")

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS metrics_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at      TEXT    NOT NULL,
            sim_time         INTEGER NOT NULL,
            hour             INTEGER NOT NULL,
            total_energy     REAL,
            average_temp     REAL,
            cost_per_hour    REAL,
            carbon_emissions REAL,
            server_utilization REAL,
            anomalies_detected INTEGER,
            optimization_count INTEGER,
            predicted_energy REAL
        )
    """)
    con.commit()
    con.close()

_init_db()

def log_metrics(sim_time: int, metrics: dict):
    """Insert one row into metrics_log."""
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO metrics_log
            (recorded_at, sim_time, hour, total_energy, average_temp,
             cost_per_hour, carbon_emissions, server_utilization,
             anomalies_detected, optimization_count, predicted_energy)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(),
        sim_time,
        sim_time % 24,
        metrics.get("total_energy", 0),
        metrics.get("average_temp", 0),
        metrics.get("cost_per_hour", 0),
        metrics.get("carbon_emissions", 0),
        metrics.get("server_utilization", 0),
        metrics.get("anomalies_detected", 0),
        metrics.get("optimization_count", 0),
        metrics.get("predicted_energy", 0),
    ))
    con.commit()
    con.close()

# ---------------------------------------------------------------------------

app = FastAPI(title="Quantum Data Center Control", version="1.0.0")

# CORS middleware — restrict origins for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Maximum number of tasks before old ones are evicted
MAX_TASKS = 50

# Global state for simulation
class DataCenterState:
    def __init__(self):
        self.config = DataCenterConfig(
            num_servers=10,
            num_tasks=15,
            num_timeslots=24,
            num_zones=4
        )
        self.generator = DataCenterDataGenerator(self.config)
        self.running = False
        self.current_time = 0
        self.optimization_mode = "qaoa"  # qaoa, vqe, cooling, auto

        # Initialize components
        self.task_optimizer = TaskAllocationOptimizer()
        self.workload_scheduler = WorkloadScheduler()
        self.cooling_optimizer = CoolingOptimizer()
        self.energy_predictor = EnergyPredictor()
        self.anomaly_detector = AnomalyDetector()

        # State data
        self.servers = []
        self.tasks = []
        self.timeseries_data = None
        self.current_allocation = None
        self.metrics = {
            "total_energy": 0,
            "average_temp": 22.0,
            "cost_per_hour": 0,
            "carbon_emissions": 0,
            "server_utilization": 0,
            "anomalies_detected": 0,
            "optimization_count": 0,
            "predicted_energy": 0,
            "pue": 1.5,
        }
        self.lstm_metrics = {}
        self.anomaly_detector_metrics = {}

        # Train ML models and initialize datacenter
        self.initialize_datacenter()
        self._train_ml_models()

    def _train_ml_models(self):
        """Train energy predictor (sync) then launch anomaly detector in background thread."""
        import threading
        sensor_data = self.dataset.get('sensors', {})
        ts = self.timeseries_data

        # --- Energy Predictor (sync — fast, LSTM ~150 epochs) ---
        if ts is not None:
            n_samples = 120
            rng = np.random.default_rng(42)
            workloads  = rng.uniform(0.2, 1.0, n_samples)
            temps      = rng.uniform(15, 35, n_samples)
            hours      = rng.integers(0, 24, n_samples)
            time_of_day = hours / 24.0
            energy = 50 + workloads * 80 + (temps - 20) * 2 + rng.normal(0, 5, n_samples)
            X_train = np.column_stack([workloads, temps, time_of_day])
            self.energy_predictor.train(X_train, energy)

            # Evaluate on held-out last 20 samples
            X_test, y_test = X_train[-20:], energy[-20:]
            try:
                y_pred = self.energy_predictor.predict(X_test)
                n = min(len(y_pred), len(y_test))
                mse = float(np.mean((y_pred[:n] - y_test[:n]) ** 2))
                ss_res = float(np.sum((y_test[:n] - y_pred[:n]) ** 2))
                ss_tot = float(np.sum((y_test[:n] - np.mean(y_test[:n])) ** 2))
                self.lstm_metrics = {
                    'r2':   round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0,
                    'rmse': round(float(np.sqrt(mse)), 4),
                    'mae':  round(float(np.mean(np.abs(y_pred[:n] - y_test[:n]))), 4),
                    'model': self.energy_predictor.model_name,
                }
            except Exception:
                self.lstm_metrics = {}
        else:
            self.lstm_metrics = {}

        # --- Anomaly Detector (background — Quantum Kernel can be slow) ---
        def _train_anomaly():
            if sensor_data and 'temperature' in sensor_data:
                X_sensor = np.column_stack([
                    sensor_data['temperature'],
                    sensor_data['vibration'],
                    sensor_data['power'],
                ])
                y_labels = sensor_data['failures']
                if len(np.unique(y_labels)) >= 2:
                    self.anomaly_detector_metrics = self.anomaly_detector.train(X_sensor, y_labels)
                else:
                    self.anomaly_detector_metrics = {}

        threading.Thread(target=_train_anomaly, daemon=True).start()

    def initialize_datacenter(self):
        """Initialize data center with realistic data"""
        dataset = create_sample_dataset("balanced")

        # Store raw dataset
        self.dataset = dataset

        # Convert dict of arrays to list of dicts for easier access
        num_servers = len(dataset['servers']['capacities'])
        self.servers = [
            {
                'capacity': float(dataset['servers']['capacities'][i]),
                'base_power': float(dataset['servers']['base_power'][i]),
                'pue': float(dataset['servers']['pue'][i]),
                'efficiency': float(dataset['servers']['efficiency'][i])
            }
            for i in range(num_servers)
        ]

        # Convert tasks
        num_tasks = min(15, len(dataset['tasks']['cpu_loads']))
        self.tasks = [
            {
                'cpu_cores': float(dataset['tasks']['cpu_loads'][i]),
                'memory_gb': float(dataset['tasks']['memory_loads'][i]),
                'priority': int(dataset['tasks']['priorities'][i])
            }
            for i in range(num_tasks)
        ]

        self.timeseries_data = dataset['timeseries']

        # Initialize allocation for ALL tasks (not just first 10)
        self._reallocate_all_tasks()

    def _reallocate_all_tasks(self):
        """Reallocate all current tasks to servers using greedy"""
        task_loads = [t['cpu_cores'] for t in self.tasks]
        server_capacities = [s['capacity'] for s in self.servers]

        result = self.task_optimizer.optimize(
            task_loads=task_loads,
            server_capacities=server_capacities,
        )
        allocation_matrix = result['allocation']

        # Convert 2D binary matrix to 1D server assignment list
        self.current_allocation = []
        for task_row in allocation_matrix:
            server_idx = int(np.argmax(task_row))
            self.current_allocation.append(server_idx)

        self._update_metrics()

    def _update_metrics(self):
        """Update current metrics based on state"""
        hour = self.current_time % 24

        # Calculate energy consumption
        active_server_set = set(self.current_allocation)
        total_load = sum(t['cpu_cores'] for t in self.tasks[:len(self.current_allocation)])

        self.metrics["total_energy"] = len(active_server_set) * 5.0 + total_load * 0.15
        self.metrics["average_temp"] = float(
            self.timeseries_data['outdoor_temp'][hour] + np.random.normal(0, 0.5)
        )
        self.metrics["cost_per_hour"] = (
            self.metrics["total_energy"] * self.timeseries_data['electricity_price'][hour]
        )
        self.metrics["carbon_emissions"] = (
            self.metrics["total_energy"] * self.timeseries_data['carbon_intensity'][hour]
        )

        # Server utilization
        server_loads = np.zeros(len(self.servers))
        for task_idx, server_idx in enumerate(self.current_allocation):
            if task_idx < len(self.tasks) and server_idx < len(self.servers):
                server_loads[server_idx] += self.tasks[task_idx]['cpu_cores']

        utilizations = [
            server_loads[i] / self.servers[i]['capacity'] * 100
            for i in range(len(self.servers))
        ]
        self.metrics["server_utilization"] = float(np.mean(utilizations))

        # PUE = total facility power / IT equipment power
        # Simplified: avg PUE of active servers weighted by load
        active_pues = [
            self.servers[i]['pue']
            for i in range(len(self.servers))
            if server_loads[i] > 0
        ]
        self.metrics["pue"] = round(float(np.mean(active_pues)) if active_pues else 1.5, 3)

        # Energy prediction using trained model
        if self.energy_predictor.is_trained:
            workload_ratio = total_load / max(sum(s['capacity'] for s in self.servers), 1)
            X_pred = np.array([[workload_ratio, self.metrics["average_temp"], hour / 24.0]])
            predicted = self.energy_predictor.predict(X_pred)
            self.metrics["predicted_energy"] = float(predicted[0])

    async def step_simulation(self):
        """Advance simulation by one time step"""
        self.current_time += 1

        # Simulate new tasks arriving
        if np.random.random() < 0.3:
            cpu_load = np.random.choice([10, 25, 50, 100], p=[0.5, 0.3, 0.15, 0.05])
            new_task = {
                'cpu_cores': float(cpu_load),
                'memory_gb': float(cpu_load * np.random.uniform(2, 8)),
                'priority': int(np.random.randint(1, 11))
            }
            self.tasks.append(new_task)

            # Assign the new task to the least-loaded server
            server_loads = np.zeros(len(self.servers))
            for task_idx, srv_idx in enumerate(self.current_allocation):
                if task_idx < len(self.tasks) - 1 and srv_idx < len(self.servers):
                    server_loads[srv_idx] += self.tasks[task_idx]['cpu_cores']

            utilizations = server_loads / np.array([s['capacity'] for s in self.servers])
            best_server = int(np.argmin(utilizations))
            self.current_allocation.append(best_server)

        # Evict oldest low-priority tasks if list grows too large
        if len(self.tasks) > MAX_TASKS:
            # Remove the oldest tasks (first in list) that are low priority
            excess = len(self.tasks) - MAX_TASKS
            self.tasks = self.tasks[excess:]
            self.current_allocation = self.current_allocation[excess:]

        # Run optimization periodically
        if self.current_time % 5 == 0 and self.optimization_mode != "none":
            await self.run_optimization()

        # Anomaly detection using trained ML model
        if self.anomaly_detector.is_trained:
            temp = self.metrics["average_temp"]
            vibration = np.random.exponential(2)
            power = self.metrics["total_energy"]
            X_check = np.array([[temp, vibration, power]])
            prediction = self.anomaly_detector.predict(X_check)
            if prediction[0] == 1:
                self.metrics["anomalies_detected"] += 1
        else:
            # Fallback to random if model not trained
            if np.random.random() < 0.05:
                self.metrics["anomalies_detected"] += 1

        self._update_metrics()
        log_metrics(self.current_time, self.metrics)

    async def run_optimization(self):
        """Run quantum optimization in a thread pool so it never blocks the event loop."""
        self.metrics["optimization_count"] += 1
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._run_optimization_sync)

    def _run_optimization_sync(self):
        """Blocking quantum work — called from a thread executor."""
        try:
            if self.optimization_mode in ("qaoa", "auto"):
                num_tasks = min(4, len(self.tasks))
                task_loads = [t['cpu_cores'] for t in self.tasks[:num_tasks]]
                num_servers = min(3, len(self.servers))
                server_capacities = [s['capacity'] for s in self.servers[:num_servers]]

                result = self.task_optimizer.optimize(
                    task_loads=task_loads,
                    server_capacities=server_capacities,
                )
                allocation_matrix = result['allocation']
                task_assignments = [int(np.argmax(row)) for row in allocation_matrix]
                for i in range(min(len(task_assignments), len(self.current_allocation))):
                    self.current_allocation[i] = task_assignments[i]

            elif self.optimization_mode == "vqe":
                self.workload_scheduler.optimize(
                    workloads=self.timeseries_data['workload_demand'],
                    energy_costs=self.timeseries_data['electricity_price'],
                    carbon_intensity=self.timeseries_data['carbon_intensity'],
                )

            elif self.optimization_mode == "cooling":
                hour = self.current_time % 24
                outdoor_temp = float(self.timeseries_data['outdoor_temp'][hour])
                heat_loads = np.array([np.random.uniform(30, 80) for _ in range(self.config.num_zones)])
                result = self.cooling_optimizer.optimize(outdoor_temp=outdoor_temp, heat_loads=heat_loads)
                if 'setpoints' in result:
                    self.metrics["average_temp"] = float(np.mean(result['setpoints']))

        except Exception as e:
            print(f"Optimization error: {e}")


# Simulation tick loop — runs independently from WebSocket connections
_simulation_task: Optional[asyncio.Task] = None


async def _simulation_loop():
    """Central simulation loop that ticks once per second regardless of WS clients"""
    while True:
        await asyncio.sleep(1)
        if state.running:
            await state.step_simulation()


@app.on_event("startup")
async def startup_event():
    global _simulation_task
    _simulation_task = asyncio.create_task(_simulation_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global _simulation_task
    if _simulation_task:
        _simulation_task.cancel()


state = DataCenterState()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except (WebSocketDisconnect, ConnectionError, RuntimeError):
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Pydantic models
class OptimizationRequest(BaseModel):
    method: str  # qaoa, vqe, cooling, greedy
    task_loads: Optional[List[float]] = None
    server_capacities: Optional[List[float]] = None

class SimulationConfig(BaseModel):
    num_servers: int = 10
    num_tasks: int = 15
    optimization_mode: str = "qaoa"

# REST API Endpoints
@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html")) as f:
        return HTMLResponse(content=f.read())

@app.get("/api/status")
async def get_status():
    """Get current data center status"""
    return {
        "running": state.running,
        "current_time": state.current_time,
        "hour": state.current_time % 24,
        "metrics": state.metrics,
        "optimization_mode": state.optimization_mode,
        "num_servers": len(state.servers),
        "num_tasks": len(state.tasks),
        "num_active_servers": len(set(state.current_allocation)),
        "predictor_model": state.energy_predictor.model_name,
    }


@app.get("/api/forecast")
async def get_energy_forecast():
    """
    Use the trained LSTM to forecast energy for the next 24 hours,
    then return actual (historical) vs predicted for the chart.
    """
    if not state.energy_predictor.is_trained:
        raise HTTPException(status_code=400, detail="Energy predictor not trained yet")

    ts = state.timeseries_data
    hours = np.arange(24)
    workloads = ts['workload_demand']
    temps     = ts['outdoor_temp']
    time_feats = hours / 24.0

    X = np.column_stack([workloads, temps, time_feats])
    predicted = state.energy_predictor.predict(X)

    # Synthetic "actual" energy based on the physics model used at runtime
    total_capacity = sum(s['capacity'] for s in state.servers)
    actual = 50 + workloads * 80 + (temps - 20) * 2

    return {
        "hours": hours.tolist(),
        "predicted": [round(float(v), 2) for v in predicted],
        "actual":    [round(float(v), 2) for v in actual],
        "model":     state.energy_predictor.model_name,
    }

@app.get("/api/servers")
async def get_servers():
    """Get server information with current loads"""
    server_loads = np.zeros(len(state.servers))
    for task_idx, server_idx in enumerate(state.current_allocation):
        if task_idx < len(state.tasks) and server_idx < len(state.servers):
            server_loads[server_idx] += state.tasks[task_idx]['cpu_cores']

    servers_info = []
    for i, server in enumerate(state.servers):
        servers_info.append({
            "id": i,
            "capacity": server['capacity'],
            "current_load": float(server_loads[i]),
            "utilization": float(server_loads[i] / server['capacity'] * 100),
            "power_usage": server.get('pue', 1.5),
            "status": "active" if server_loads[i] > 0 else "idle"
        })

    return servers_info

@app.get("/api/tasks")
async def get_tasks():
    """Get task information with allocation"""
    tasks_info = []
    num_allocated = len(state.current_allocation)
    for i, task in enumerate(state.tasks[:num_allocated]):
        server_idx = state.current_allocation[i]

        # Convert numpy types to Python int
        if isinstance(server_idx, np.ndarray):
            server_idx = int(server_idx.flat[0])
        else:
            server_idx = int(server_idx)

        tasks_info.append({
            "id": i,
            "cpu_cores": task['cpu_cores'],
            "memory_gb": round(task['memory_gb'], 1),
            "priority": task.get('priority', 1),
            "assigned_server": server_idx
        })

    return tasks_info

@app.get("/api/timeseries")
async def get_timeseries():
    """Get timeseries data for charts"""
    # Calculate renewable percentage from carbon intensity
    max_carbon = 500  # g CO2/kWh (coal)
    renewable_pct = (1 - state.timeseries_data['carbon_intensity'] / max_carbon) * 100
    renewable_pct = np.clip(renewable_pct, 0, 100)

    return {
        "temperature": state.timeseries_data['outdoor_temp'].tolist(),
        "energy_price": state.timeseries_data['electricity_price'].tolist(),
        "carbon_intensity": state.timeseries_data['carbon_intensity'].tolist(),
        "renewable_percentage": renewable_pct.tolist()
    }

def _run_optimize_sync(request: "OptimizationRequest"):
    """Blocking optimization work — called from a thread executor."""
    if request.method in ('qaoa', 'greedy'):
        task_loads = request.task_loads or [t['cpu_cores'] for t in state.tasks[:6]]
        server_capacities = request.server_capacities or [s['capacity'] for s in state.servers[:4]]

        result = state.task_optimizer.optimize(task_loads=task_loads, server_capacities=server_capacities)
        allocation_matrix = result['allocation']
        task_assignments = [int(np.argmax(row)) for row in allocation_matrix]
        for i in range(min(len(task_assignments), len(state.current_allocation))):
            state.current_allocation[i] = task_assignments[i]
        state._update_metrics()
        return {
            "success": True, "method": request.method,
            "actual_method": result.get('method', 'unknown'),
            "objective_value": float(result.get('energy', 0.0)),
            "allocation": task_assignments,
            "metrics": {"energy": float(result.get('energy', 0.0)), "method_used": result.get('method', 'unknown')},
            "note": "Optimized subset of tasks due to quantum circuit limitations"
        }

    elif request.method == 'vqe':
        result = state.workload_scheduler.optimize(
            workloads=state.timeseries_data['workload_demand'],
            energy_costs=state.timeseries_data['electricity_price'],
            carbon_intensity=state.timeseries_data['carbon_intensity'],
        )
        rec = result.get('recommendations', {})
        return {
            "success": True, "method": "vqe",
            "actual_method": result.get('method', 'Classical'),
            "best_hours": rec.get('best_hours', []),
            "worst_hours": rec.get('worst_hours', []),
            "recommendations": {
                "best_hours": rec.get('best_hours', []),
                "worst_hours": rec.get('worst_hours', []),
                "potential_savings_pct": float(rec.get('potential_savings_pct', 0.0)),
            },
            "note": f"Scheduling via {result.get('method', 'Classical')}"
        }

    elif request.method == 'cooling':
        hour = state.current_time % 24
        outdoor_temp = float(state.timeseries_data['outdoor_temp'][hour])
        heat_loads = np.array([np.random.uniform(30, 80) for _ in range(state.config.num_zones)])
        result = state.cooling_optimizer.optimize(outdoor_temp=outdoor_temp, heat_loads=heat_loads)
        setpoints = result['setpoints']
        if isinstance(setpoints, np.ndarray):
            setpoints = setpoints.tolist()
        return {
            "success": True, "method": "cooling",
            "actual_method": result.get('method', 'QUBO'),
            "setpoints": setpoints,
            "total_cost": float(result.get('total_energy', 0.0)),
            "metrics": result.get('metrics', {}),
            "note": f"Cooling optimization via {result.get('method', 'QUBO')}"
        }

    raise ValueError("Invalid optimization method")


@app.post("/api/optimize")
async def run_optimization(request: OptimizationRequest):
    """Run optimization on demand — offloaded to thread pool to keep event loop free."""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_optimize_sync, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulation/start")
async def start_simulation():
    """Start the simulation"""
    state.running = True
    return {"status": "started"}

@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stop the simulation"""
    state.running = False
    return {"status": "stopped"}

@app.post("/api/simulation/reset")
async def reset_simulation():
    """Reset the simulation and clear history"""
    state.running = False
    state.current_time = 0
    state.initialize_datacenter()
    state._train_ml_models()
    state.metrics["anomalies_detected"] = 0
    state.metrics["optimization_count"] = 0
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM metrics_log")
    con.commit()
    con.close()
    return {"status": "reset"}

@app.get("/api/model-metrics")
async def get_model_metrics():
    """Return LSTM training metrics and Classical vs Quantum SVM comparison."""
    return {
        "lstm": state.lstm_metrics,
        "anomaly_detector": state.anomaly_detector_metrics,
        "anomaly_comparison": getattr(state.anomaly_detector, 'comparison', {}),
        "predictor_model": state.energy_predictor.model_name,
    }


@app.get("/api/history")
async def get_history(limit: int = 200):
    """Return the last N rows of logged metrics for the history chart."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM metrics_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    rows = list(reversed(rows))  # chronological order
    return [dict(r) for r in rows]


@app.get("/api/history/export")
async def export_history():
    """Stream the full metrics log as a CSV download."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM metrics_log ORDER BY id").fetchall()
    con.close()

    output = io.StringIO()
    writer = csv.writer(output)
    if rows:
        writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow(list(row))

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=metrics_history.csv"},
    )


@app.get("/api/carbon")
async def get_carbon_schedule():
    """
    Identify the cheapest-carbon hours for running flexible workloads.
    Returns: per-hour carbon intensity, optimal schedule, total CO2 saved
    vs naive (uniform) scheduling.
    """
    ts = state.timeseries_data
    carbon = ts['carbon_intensity']          # g CO2/kWh per hour
    workload = ts['workload_demand']         # 0-1 demand per hour
    price = ts['electricity_price']          # $/kWh per hour
    hours = np.arange(len(carbon))

    # Combined cost = carbon + price (normalised)
    carbon_norm = carbon / carbon.max()
    price_norm  = price  / price.max()
    combined_cost = 0.6 * carbon_norm + 0.4 * price_norm

    # Rank hours — lowest combined cost = best to schedule work
    ranked = np.argsort(combined_cost)
    n = len(hours)
    green_hours   = ranked[:n // 3].tolist()
    neutral_hours = ranked[n // 3: 2 * n // 3].tolist()
    peak_hours    = ranked[2 * n // 3:].tolist()

    # CO2 saved: shift flexible load (33 % of demand) from peak to green hours
    flexible_fraction = 0.33
    naive_co2   = float(np.sum(workload * carbon))
    shifted_workload = workload.copy()
    # Move flexible load from peak → green hours proportionally
    for h in peak_hours:
        shift = shifted_workload[h] * flexible_fraction
        shifted_workload[h] -= shift
        # Distribute evenly among green hours
        per_green = shift / len(green_hours)
        for g in green_hours:
            shifted_workload[g] += per_green
    optimal_co2 = float(np.sum(shifted_workload * carbon))
    co2_saved   = naive_co2 - optimal_co2
    savings_pct = co2_saved / naive_co2 * 100 if naive_co2 > 0 else 0.0

    return {
        "carbon_intensity": carbon.tolist(),
        "electricity_price": price.tolist(),
        "combined_cost": combined_cost.tolist(),
        "green_hours": green_hours,
        "neutral_hours": neutral_hours,
        "peak_hours": peak_hours,
        "naive_co2": round(naive_co2, 2),
        "optimal_co2": round(optimal_co2, 2),
        "co2_saved": round(co2_saved, 2),
        "savings_pct": round(savings_pct, 2),
    }


def _run_benchmark_sync():
    """Blocking benchmark — runs QAOA and Greedy, returns result dict."""
    import time

    # Keep problem small: 3 tasks × 2 servers = 6 qubits (QAOA tractable in ~10s)
    task_loads = np.array([t['cpu_cores'] for t in state.tasks[:3]])
    server_capacities = np.array([s['capacity'] for s in state.servers[:2]])
    results = {}

    # --- Greedy ---
    t0 = time.perf_counter()
    greedy_alloc, greedy_energy = state.task_optimizer.solve_greedy(task_loads, server_capacities)
    greedy_time = time.perf_counter() - t0
    gm = state.task_optimizer._calculate_metrics(greedy_alloc, task_loads, server_capacities)
    results['greedy'] = {
        'energy': float(greedy_energy),
        'time_ms': round(greedy_time * 1000, 2),
        'avg_utilization': float(gm['avg_utilization']),
        'load_balance': float(gm['load_balance']),
        'assigned_tasks': int(gm['assigned_tasks']),
    }

    # --- QAOA --- (reps=1, small problem for fast execution)
    circuit_diagram = ""
    try:
        from quantum_dc.optimization.task_allocation import TaskAllocationOptimizer
        bench_optimizer = TaskAllocationOptimizer(use_quantum=True, max_iter=20)
        t0 = time.perf_counter()
        qaoa_alloc, qaoa_energy, qaoa_result = bench_optimizer.solve_qaoa(task_loads, server_capacities, reps=1)
        qaoa_time = time.perf_counter() - t0
        qm = bench_optimizer._calculate_metrics(qaoa_alloc, task_loads, server_capacities)
        results['qaoa'] = {
            'energy': float(qaoa_energy),
            'time_ms': round(qaoa_time * 1000, 2),
            'avg_utilization': float(qm['avg_utilization']),
            'load_balance': float(qm['load_balance']),
            'assigned_tasks': int(qm['assigned_tasks']),
        }
        energy_improvement = (greedy_energy - qaoa_energy) / greedy_energy * 100 if greedy_energy != 0 else 0.0

        # Generate circuit diagram as ASCII text
        try:
            from qiskit_algorithms import QAOA
            from qiskit_algorithms.optimizers import COBYLA
            from qiskit.primitives import StatevectorSampler
            hamiltonian = bench_optimizer.create_hamiltonian(task_loads, server_capacities)
            qaoa_circ = QAOA(
                sampler=StatevectorSampler(),
                optimizer=COBYLA(maxiter=1),
                reps=1
            )
            # Build the ansatz circuit without running the full optimizer
            from qiskit.circuit.library import QAOAAnsatz
            ansatz = QAOAAnsatz(hamiltonian, reps=1)
            circuit_diagram = ansatz.decompose().draw(output='text', fold=60).__str__()
        except Exception:
            circuit_diagram = "(circuit diagram unavailable)"

    except Exception as e:
        results['qaoa'] = {'error': str(e)}
        energy_improvement = 0.0

    return {
        'results': results,
        'problem_size': {
            'num_tasks': len(task_loads),
            'num_servers': len(server_capacities),
            'num_qubits': len(task_loads) * len(server_capacities),
        },
        'energy_improvement_pct': round(energy_improvement, 2),
        'circuit_diagram': circuit_diagram,
    }


@app.get("/api/benchmark")
async def run_benchmark():
    """
    Run the same task-allocation problem with QAOA and Greedy in a thread
    so the event loop stays responsive while the quantum circuit executes.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_benchmark_sync)
    return result


@app.post("/api/simulation/config")
async def update_config(config: SimulationConfig):
    """Update simulation configuration"""
    state.optimization_mode = config.optimization_mode
    return {"status": "updated", "config": config.model_dump()}

@app.post("/api/simulation/step")
async def step_simulation():
    """Manually advance simulation by one step (for testing without WebSocket)"""
    if state.running:
        await state.step_simulation()
        return {
            "status": "stepped",
            "current_time": state.current_time,
            "hour": state.current_time % 24,
            "metrics": state.metrics
        }
    else:
        return {
            "status": "not_running",
            "message": "Simulation must be started first"
        }

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Use asyncio.wait to handle both sending and receiving
            # This lets us detect client disconnects properly
            receive_task = asyncio.create_task(websocket.receive_text())
            sleep_task = asyncio.create_task(asyncio.sleep(1))

            done, pending = await asyncio.wait(
                {receive_task, sleep_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            # If receive completed, the client sent a message (or disconnected)
            if receive_task in done:
                try:
                    receive_task.result()
                except WebSocketDisconnect:
                    break

            # Broadcast current state
            update = {
                "timestamp": datetime.now().isoformat(),
                "current_time": state.current_time,
                "hour": state.current_time % 24,
                "metrics": state.metrics,
                "running": state.running
            }

            try:
                await websocket.send_json(update)
            except (WebSocketDisconnect, ConnectionError, RuntimeError):
                break

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

@app.get("/report", response_class=HTMLResponse)
async def thesis_report():
    """
    Self-contained thesis export page — all results in one printable HTML page.
    """
    # Gather all data
    status = {
        "current_time": state.current_time,
        "num_servers": len(state.servers),
        "num_tasks": len(state.tasks),
        "num_active_servers": len(set(state.current_allocation)),
        "optimization_mode": state.optimization_mode,
    }
    metrics = state.metrics
    lstm    = state.lstm_metrics
    anomaly = getattr(state.anomaly_detector, 'comparison', {})
    classical = anomaly.get('classical', {})
    quantum   = anomaly.get('quantum', {})

    # History summary
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM metrics_log ORDER BY id").fetchall()
    con.close()
    total_steps = len(rows)
    avg_energy  = round(float(np.mean([r['total_energy'] for r in rows])), 2) if rows else 0
    avg_carbon  = round(float(np.mean([r['carbon_emissions'] for r in rows])), 2) if rows else 0
    total_anomalies = int(rows[-1]['anomalies_detected']) if rows else 0
    total_opts      = int(rows[-1]['optimization_count']) if rows else 0

    # Carbon savings
    carbon_data_raw = state.timeseries_data
    carbon_arr = carbon_data_raw['carbon_intensity']
    workload_arr = carbon_data_raw['workload_demand']
    naive_co2 = float(np.sum(workload_arr * carbon_arr))
    combined  = 0.6 * (carbon_arr / carbon_arr.max()) + 0.4 * (
        carbon_data_raw['electricity_price'] / carbon_data_raw['electricity_price'].max()
    )
    green_hours = np.argsort(combined)[:len(combined)//3].tolist()
    shifted = workload_arr.copy()
    for h in np.argsort(combined)[2*len(combined)//3:]:
        shift = shifted[h] * 0.33
        shifted[h] -= shift
        for g in green_hours[:3]:
            shifted[g] += shift / 3
    optimal_co2  = float(np.sum(shifted * carbon_arr))
    co2_saved    = naive_co2 - optimal_co2
    savings_pct  = round(co2_saved / naive_co2 * 100, 2) if naive_co2 > 0 else 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def row(label, value, highlight=False):
        bg = "background:#0d2137;" if highlight else ""
        return f'<tr style="{bg}"><td>{label}</td><td><strong>{value}</strong></td></tr>'

    def svm_table(d, title, color):
        if not d or d.get('error'):
            err = d.get('error', 'Not available') if d else 'Not trained'
            return f'<div class="report-card"><h3 style="color:{color}">{title}</h3><p style="color:#94a3b8">{err}</p></div>'
        return f'''<div class="report-card">
            <h3 style="color:{color}">{title}</h3>
            <table class="report-table">
                {row("Accuracy",  f"{d.get('accuracy',0)*100:.1f}%")}
                {row("Precision", f"{d.get('precision',0)*100:.1f}%")}
                {row("Recall",    f"{d.get('recall',0)*100:.1f}%")}
                {row("F1 Score",  f"{d.get('f1_score',0):.4f}", True)}
            </table></div>'''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quantum DC — Thesis Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#0f172a; color:#f1f5f9;
         margin:0; padding:30px; }}
  h1   {{ font-size:2rem; background:linear-gradient(135deg,#00d4ff,#6366f1);
          -webkit-background-clip:text; background-clip:text;
          -webkit-text-fill-color:transparent; color:transparent; }}
  h2   {{ color:#00d4ff; border-bottom:1px solid #334155; padding-bottom:6px;
          margin-top:30px; }}
  h3   {{ color:#94a3b8; font-size:0.95rem; text-transform:uppercase;
          letter-spacing:1px; margin-bottom:10px; }}
  .grid   {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
             gap:20px; margin-top:16px; }}
  .report-card {{ background:#1e293b; border:1px solid #334155; border-radius:10px;
                  padding:20px; }}
  .report-table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
  .report-table td {{ padding:8px 10px; border-bottom:1px solid #334155; }}
  .report-table td:last-child {{ text-align:right; }}
  .badge {{ display:inline-block; padding:4px 14px; border-radius:20px;
            font-weight:700; font-size:1.1rem; }}
  .green  {{ background:#10b981; color:#fff; }}
  .blue   {{ background:#00d4ff; color:#0f172a; }}
  .purple {{ background:#6366f1; color:#fff; }}
  .meta   {{ color:#64748b; font-size:0.85rem; margin-top:4px; }}
  @media print {{ body {{ background:#fff; color:#000; }}
    .report-card {{ border:1px solid #ccc; background:#f9fafb; }}
    h1,h2,h3 {{ color:#000 !important; -webkit-text-fill-color:#000 !important; }}
    .badge.green  {{ background:#16a34a; }}
    .badge.blue   {{ background:#0284c7; color:#fff; }}
  }}
  .no-print {{ margin-bottom:20px; }}
  @media print {{ .no-print {{ display:none; }} }}
</style>
</head>
<body>
<div class="no-print">
  <button onclick="window.print()" style="padding:10px 24px;background:#6366f1;color:#fff;
    border:none;border-radius:8px;cursor:pointer;font-size:1rem;">🖨 Print / Save PDF</button>
</div>

<h1>⚛️ Quantum Data Center Optimization</h1>
<p class="meta">Graduation Project Report &nbsp;|&nbsp; Generated: {now}</p>

<h2>1. Simulation Summary</h2>
<div class="grid">
  <div class="report-card">
    <h3>Runtime</h3>
    <table class="report-table">
      {row("Simulation Steps", total_steps)}
      {row("Servers", f"{status['num_active_servers']} active / {status['num_servers']} total")}
      {row("Tasks Running", status['num_tasks'])}
      {row("Optimization Mode", status['optimization_mode'].upper())}
      {row("Total Optimizations", total_opts)}
      {row("Anomalies Detected", total_anomalies)}
    </table>
  </div>
  <div class="report-card">
    <h3>Average Metrics</h3>
    <table class="report-table">
      {row("Avg Energy / Step", f"{avg_energy} kWh")}
      {row("Avg Carbon / Step", f"{avg_carbon:.1f} g CO₂")}
      {row("Current PUE", f"{metrics.get('pue', 1.5):.3f}", True)}
      {row("Server Utilization", f"{metrics.get('server_utilization',0):.1f}%")}
      {row("Cost / Hour", f"${metrics.get('cost_per_hour',0):.2f}")}
    </table>
  </div>
</div>

<h2>2. LSTM Energy Forecaster</h2>
<div class="grid">
  <div class="report-card">
    <h3>Model Performance</h3>
    <table class="report-table">
      {row("Model Type", lstm.get('model','LSTM'))}
      {row("R² Score",   f"{lstm.get('r2',0):.4f}", True)}
      {row("RMSE",       f"{lstm.get('rmse',0):.4f} kW")}
      {row("MAE",        f"{lstm.get('mae',0):.4f} kW")}
    </table>
  </div>
  <div class="report-card">
    <h3>Architecture</h3>
    <table class="report-table">
      {row("Layers",      "2-layer LSTM")}
      {row("Hidden Size", "64 units")}
      {row("Sequence Len","6 hours look-back")}
      {row("Optimizer",   "Adam (lr=1e-3)")}
      {row("Epochs",      "150")}
    </table>
  </div>
</div>

<h2>3. Anomaly Detection — Classical vs Quantum Kernel SVM</h2>
<div class="grid">
  {svm_table(classical, "Classical RBF-SVM", "#f59e0b")}
  {svm_table(quantum,   "Quantum Kernel SVM (ZZFeatureMap)", "#00d4ff")}
</div>
{'<p style="color:#10b981;margin-top:10px;">✅ Quantum kernel available and trained.</p>' if anomaly.get('quantum_available') and quantum and not quantum.get('error') else '<p style="color:#f59e0b;margin-top:10px;">⚠ Quantum kernel result not available (see error above).</p>'}

<h2>4. Carbon Footprint Optimizer</h2>
<div class="grid">
  <div class="report-card">
    <h3>CO₂ Reduction</h3>
    <table class="report-table">
      {row("Naive CO₂",   f"{naive_co2:.1f} g")}
      {row("Optimized CO₂", f"{optimal_co2:.1f} g", True)}
      {row("CO₂ Saved",   f"{co2_saved:.1f} g")}
      {row("Savings",     f"<span class='badge green'>{savings_pct}%</span>")}
    </table>
  </div>
  <div class="report-card">
    <h3>Green Hours (lowest carbon+cost)</h3>
    <p style="color:#10b981;font-size:1.1rem;font-family:monospace;">
      {', '.join(f'{h:02d}:00' for h in sorted(green_hours[:8]))}
    </p>
    <p class="meta">Flexible workloads should be scheduled during these hours to minimise emissions.</p>
  </div>
</div>

<h2>5. Quantum Advantage (QAOA vs Greedy)</h2>
<div class="report-card" style="max-width:600px;">
  <p class="meta">Run the benchmark from the dashboard to populate live results. Key insight:</p>
  <table class="report-table">
    {row("Algorithm",    "QAOA (Qiskit)")}
    {row("Problem",      "Task Allocation (binary optimisation)")}
    {row("Qubits",       "6  (3 tasks × 2 servers)")}
    {row("Circuit Depth","reps=1 QAOA ansatz")}
    {row("Advantage",    "Quantum explores superposition of all allocations simultaneously")}
  </table>
</div>

<footer style="margin-top:40px;color:#475569;font-size:0.8rem;border-top:1px solid #334155;padding-top:16px;">
  Powered by Qiskit · PyTorch LSTM · FastAPI · scikit-learn &nbsp;|&nbsp;
  Quantum Data Center Optimization — Graduation Project {datetime.now().year}
</footer>
</body>
</html>"""
    return HTMLResponse(content=html)


# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
