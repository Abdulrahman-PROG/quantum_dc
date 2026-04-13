"""
FastAPI Backend for Quantum Data Center Optimization
Real-time monitoring and control interface
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import numpy as np
from datetime import datetime
import sys
import os

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
        }

        # Train ML models and initialize datacenter
        self.initialize_datacenter()
        self._train_ml_models()

    def _train_ml_models(self):
        """Train the energy predictor and anomaly detector on synthetic data"""
        # --- Energy Predictor ---
        sensor_data = self.dataset.get('sensors', {})
        ts = self.timeseries_data
        if ts is not None:
            n_samples = 100
            rng = np.random.default_rng(42)
            workloads = rng.uniform(0.2, 1.0, n_samples)
            temps = rng.uniform(15, 35, n_samples)
            hours = rng.integers(0, 24, n_samples)
            time_of_day = hours / 24.0
            # Synthetic energy: base + workload contribution + temp contribution
            energy = 50 + workloads * 80 + (temps - 20) * 2 + rng.normal(0, 5, n_samples)
            X_train = np.column_stack([workloads, temps, time_of_day])
            self.energy_predictor.train(X_train, energy)

        # --- Anomaly Detector ---
        if sensor_data and 'temperature' in sensor_data:
            X_sensor = np.column_stack([
                sensor_data['temperature'],
                sensor_data['vibration'],
                sensor_data['power'],
            ])
            y_labels = sensor_data['failures']
            # SVM requires at least 2 classes — skip training if only 1 class
            if len(np.unique(y_labels)) >= 2:
                self.anomaly_detector.train(X_sensor, y_labels)

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

    async def run_optimization(self):
        """Run quantum optimization based on mode"""
        self.metrics["optimization_count"] += 1

        try:
            if self.optimization_mode in ("qaoa", "auto"):
                # Task allocation optimization
                num_tasks = min(4, len(self.tasks))
                task_loads = [t['cpu_cores'] for t in self.tasks[:num_tasks]]
                num_servers = min(3, len(self.servers))
                server_capacities = [s['capacity'] for s in self.servers[:num_servers]]

                # Let the optimizer decide: QAOA if small enough, greedy otherwise
                result = self.task_optimizer.optimize(
                    task_loads=task_loads,
                    server_capacities=server_capacities,
                )

                allocation_matrix = result['allocation']
                task_assignments = []
                for task_row in allocation_matrix:
                    server_idx = int(np.argmax(task_row))
                    task_assignments.append(server_idx)

                for i in range(min(len(task_assignments), len(self.current_allocation))):
                    self.current_allocation[i] = task_assignments[i]

            elif self.optimization_mode == "vqe":
                # Workload scheduling
                workloads = self.timeseries_data['workload_demand']
                energy_costs = self.timeseries_data['electricity_price']
                carbon_intensity = self.timeseries_data['carbon_intensity']

                self.workload_scheduler.optimize(
                    workloads=workloads,
                    energy_costs=energy_costs,
                    carbon_intensity=carbon_intensity,
                )

            elif self.optimization_mode == "cooling":
                # Cooling optimization using QUBO
                hour = self.current_time % 24
                outdoor_temp = float(self.timeseries_data['outdoor_temp'][hour])
                heat_loads = np.array([
                    np.random.uniform(30, 80) for _ in range(self.config.num_zones)
                ])

                result = self.cooling_optimizer.optimize(
                    outdoor_temp=outdoor_temp,
                    heat_loads=heat_loads,
                )

                # Apply optimal temperature as the average setpoint
                if 'setpoints' in result:
                    avg_setpoint = float(np.mean(result['setpoints']))
                    self.metrics["average_temp"] = avg_setpoint

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
        "num_active_servers": len(set(state.current_allocation))
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

@app.post("/api/optimize")
async def run_optimization(request: OptimizationRequest):
    """Run optimization on demand"""
    try:
        if request.method in ('qaoa', 'greedy'):
            task_loads = request.task_loads or [t['cpu_cores'] for t in state.tasks[:6]]
            server_capacities = request.server_capacities or [
                s['capacity'] for s in state.servers[:4]
            ]

            # Let the optimizer decide QAOA vs greedy based on problem size
            result = state.task_optimizer.optimize(
                task_loads=task_loads,
                server_capacities=server_capacities,
            )

            allocation_matrix = result['allocation']
            task_assignments = []
            for task_row in allocation_matrix:
                server_idx = int(np.argmax(task_row))
                task_assignments.append(server_idx)

            # Update allocation for the optimized tasks
            for i in range(min(len(task_assignments), len(state.current_allocation))):
                state.current_allocation[i] = task_assignments[i]

            state._update_metrics()

            return {
                "success": True,
                "method": request.method,
                "actual_method": result.get('method', 'unknown'),
                "objective_value": float(result.get('energy', 0.0)),
                "allocation": task_assignments,
                "metrics": {
                    "energy": float(result.get('energy', 0.0)),
                    "method_used": result.get('method', 'unknown')
                },
                "note": "Optimized subset of tasks due to quantum circuit limitations"
            }

        elif request.method == 'vqe':
            # Workload scheduling via the actual WorkloadScheduler
            workloads = state.timeseries_data['workload_demand']
            energy_prices = state.timeseries_data['electricity_price']
            carbon_intensities = state.timeseries_data['carbon_intensity']

            result = state.workload_scheduler.optimize(
                workloads=workloads,
                energy_costs=energy_prices,
                carbon_intensity=carbon_intensities,
            )

            recommendations = result.get('recommendations', {})
            potential_savings = recommendations.get('potential_savings_pct', 0.0)

            return {
                "success": True,
                "method": "vqe",
                "actual_method": result.get('method', 'Classical'),
                "best_hours": recommendations.get('best_hours', []),
                "worst_hours": recommendations.get('worst_hours', []),
                "recommendations": {
                    "best_hours": recommendations.get('best_hours', []),
                    "worst_hours": recommendations.get('worst_hours', []),
                    "potential_savings_pct": float(potential_savings),
                },
                "note": f"Scheduling via {result.get('method', 'Classical')}"
            }

        elif request.method == 'cooling':
            # Cooling optimization via the actual CoolingOptimizer
            hour = state.current_time % 24
            outdoor_temp = float(state.timeseries_data['outdoor_temp'][hour])
            heat_loads = np.array([
                np.random.uniform(30, 80) for _ in range(state.config.num_zones)
            ])

            result = state.cooling_optimizer.optimize(
                outdoor_temp=outdoor_temp,
                heat_loads=heat_loads,
            )

            setpoints = result['setpoints']
            if isinstance(setpoints, np.ndarray):
                setpoints = setpoints.tolist()

            return {
                "success": True,
                "method": "cooling",
                "actual_method": result.get('method', 'QUBO'),
                "setpoints": setpoints,
                "total_cost": float(result.get('total_energy', 0.0)),
                "metrics": result.get('metrics', {}),
                "note": f"Cooling optimization via {result.get('method', 'QUBO')}"
            }

        else:
            raise HTTPException(status_code=400, detail="Invalid optimization method")

    except HTTPException:
        raise
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
    """Reset the simulation"""
    state.running = False
    state.current_time = 0
    state.initialize_datacenter()
    state._train_ml_models()
    state.metrics["anomalies_detected"] = 0
    state.metrics["optimization_count"] = 0
    return {"status": "reset"}

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

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
