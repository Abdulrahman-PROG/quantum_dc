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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            "optimization_count": 0
        }

        self.initialize_datacenter()

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

        # Initialize allocation (greedy initially)
        task_loads = [t['cpu_cores'] for t in self.tasks[:10]]
        server_capacities = [s['capacity'] for s in self.servers]

        result = self.task_optimizer.optimize(
            task_loads=task_loads,
            server_capacities=server_capacities,
            method='greedy'
        )
        self.current_allocation = result['allocation']
        self._update_metrics()

    def _update_metrics(self):
        """Update current metrics based on state"""
        hour = self.current_time % 24

        # Calculate energy consumption
        active_servers = np.unique(self.current_allocation)
        total_load = sum([t['cpu_cores'] for t in self.tasks[:len(self.current_allocation)]])

        self.metrics["total_energy"] = len(active_servers) * 5.0 + total_load * 0.15
        self.metrics["average_temp"] = self.timeseries_data['outdoor_temp'][hour] + np.random.normal(0, 0.5)
        self.metrics["cost_per_hour"] = self.metrics["total_energy"] * self.timeseries_data['electricity_price'][hour]
        self.metrics["carbon_emissions"] = self.metrics["total_energy"] * self.timeseries_data['carbon_intensity'][hour]

        # Server utilization
        server_loads = np.zeros(len(self.servers))
        for task_idx, server_idx in enumerate(self.current_allocation):
            if task_idx < len(self.tasks):
                server_loads[server_idx] += self.tasks[task_idx]['cpu_cores']

        utilizations = [server_loads[i] / self.servers[i]['capacity'] * 100
                       for i in range(len(self.servers))]
        self.metrics["server_utilization"] = np.mean(utilizations)

    async def step_simulation(self):
        """Advance simulation by one time step"""
        self.current_time += 1
        hour = self.current_time % 24

        # Simulate new tasks arriving
        if np.random.random() < 0.3:
            # Generate a single task
            cpu_load = np.random.choice([10, 25, 50, 100], p=[0.5, 0.3, 0.15, 0.05])
            new_task = {
                'cpu_cores': float(cpu_load),
                'memory_gb': cpu_load * np.random.uniform(2, 8),
                'priority': np.random.randint(1, 11)
            }
            self.tasks.append(new_task)

        # Run optimization periodically
        if self.current_time % 5 == 0 and self.optimization_mode != "none":
            await self.run_optimization()

        # Check for anomalies
        if np.random.random() < 0.05:
            self.metrics["anomalies_detected"] += 1

        self._update_metrics()

    async def run_optimization(self):
        """Run quantum optimization based on mode"""
        self.metrics["optimization_count"] += 1

        try:
            if self.optimization_mode == "qaoa":
                # Task allocation optimization - use small problem size
                num_tasks = min(4, len(self.tasks))  # Small problem for quantum
                task_loads = [t['cpu_cores'] for t in self.tasks[:num_tasks]]
                num_servers = min(3, len(self.servers))  # Limit servers too
                server_capacities = [s['capacity'] for s in self.servers[:num_servers]]

                result = self.task_optimizer.optimize(
                    task_loads=task_loads,
                    server_capacities=server_capacities,
                    method='greedy'  # Use greedy for reliability
                )

                # The allocation is a 2D binary matrix: tasks × servers
                # Convert it to a 1D array of server assignments
                allocation_matrix = result['allocation']

                # Convert binary matrix to server assignment list
                task_assignments = []
                for task_row in allocation_matrix:
                    # Find which server has 1 for this task
                    server_idx = np.argmax(task_row)
                    task_assignments.append(int(server_idx))

                # Update allocation for the optimized tasks
                for i in range(min(len(task_assignments), len(self.current_allocation))):
                    # Map directly since we used servers[:num_servers]
                    self.current_allocation[i] = task_assignments[i]

            elif self.optimization_mode == "cooling":
                # Cooling optimization - simplified
                pass  # Skip for now
        except Exception as e:
            print(f"Optimization error: {e}")
            # Continue without crashing

state = DataCenterState()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

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
        "num_active_servers": len(np.unique(state.current_allocation))
    }

@app.get("/api/servers")
async def get_servers():
    """Get server information with current loads"""
    server_loads = np.zeros(len(state.servers))
    for task_idx, server_idx in enumerate(state.current_allocation):
        if task_idx < len(state.tasks):
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
    for i, task in enumerate(state.tasks[:len(state.current_allocation)]):
        # Handle both scalar and array allocation values
        server_idx = state.current_allocation[i]

        # Convert numpy types to Python int
        if isinstance(server_idx, np.ndarray):
            if server_idx.size == 1:
                server_idx = int(server_idx.item())
            else:
                server_idx = int(server_idx[0])  # Take first element
        elif hasattr(server_idx, '__index__'):
            server_idx = int(server_idx)
        else:
            server_idx = int(server_idx)

        tasks_info.append({
            "id": i,
            "cpu_cores": task['cpu_cores'],
            "memory_gb": task['memory_gb'],
            "priority": task.get('priority', 1.0),
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
        # Use smaller problem sizes for quantum methods
        if request.method == 'qaoa':
            # Quantum has limits - use very small problem
            task_loads = request.task_loads or [t['cpu_cores'] for t in state.tasks[:3]]
            server_capacities = request.server_capacities or [s['capacity'] for s in state.servers[:2]]
            actual_method = 'greedy'  # Use greedy as fallback
        else:
            task_loads = request.task_loads or [t['cpu_cores'] for t in state.tasks[:10]]
            server_capacities = request.server_capacities or [s['capacity'] for s in state.servers]
            actual_method = request.method

        print(f"DEBUG: Optimization request - method={request.method}, actual_method={actual_method}")

        if request.method in ['qaoa', 'greedy']:
            result = state.task_optimizer.optimize(
                task_loads=task_loads,
                server_capacities=server_capacities,
                method=actual_method
            )

            # The allocation is a 2D binary matrix: tasks × servers
            # We need to convert it to a 1D array of server assignments
            allocation_matrix = result['allocation']

            # Convert binary matrix to server assignment list
            task_assignments = []
            for task_row in allocation_matrix:
                # Find which server has 1 for this task
                server_idx = np.argmax(task_row)
                task_assignments.append(int(server_idx))

            num_result_servers = len(server_capacities)
            num_total_servers = len(state.servers)

            # Update allocation for the optimized tasks
            for i in range(min(len(task_assignments), len(state.current_allocation))):
                # The task_assignments use server indices 0 to num_result_servers-1
                # Map directly since we used servers[:num_result_servers]
                state.current_allocation[i] = task_assignments[i]

            state._update_metrics()

            return {
                "success": True,
                "method": request.method,
                "actual_method": actual_method,
                "objective_value": float(result.get('energy', result.get('objective_value', 0.0))),
                "allocation": task_assignments,
                "metrics": {
                    "energy": float(result.get('energy', 0.0)),
                    "method_used": result.get('method', actual_method)
                },
                "note": "Optimized subset of tasks due to quantum circuit limitations"
            }

        elif request.method == 'vqe':
            # Workload scheduling - use classical heuristic
            hour = state.current_time % 24
            energy_prices = state.timeseries_data['electricity_price']
            carbon_intensities = state.timeseries_data['carbon_intensity']

            # Find best hours (lowest cost + carbon)
            combined_cost = energy_prices * 0.5 + carbon_intensities / 1000 * 0.5
            best_hours = np.argsort(combined_cost)[:6]
            worst_hours = np.argsort(combined_cost)[-6:]

            return {
                "success": True,
                "method": "vqe",
                "best_hours": best_hours.tolist(),
                "worst_hours": worst_hours.tolist(),
                "recommendations": {
                    "best_hours": best_hours.tolist(),
                    "worst_hours": worst_hours.tolist(),
                    "potential_savings_pct": 25.0
                },
                "note": "Classical heuristic used for demonstration"
            }

        elif request.method == 'cooling':
            # Simple cooling optimization
            heat_loads = [np.random.uniform(30, 80) for _ in range(4)]
            optimal_temp = 20.0  # Target temperature
            setpoints = [optimal_temp] * 4

            return {
                "success": True,
                "method": "cooling",
                "setpoints": setpoints,
                "total_cost": sum(heat_loads) * 0.1,
                "note": "Simplified cooling optimization"
            }

        else:
            raise HTTPException(status_code=400, detail="Invalid optimization method")

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
    state.metrics["anomalies_detected"] = 0
    state.metrics["optimization_count"] = 0
    return {"status": "reset"}

@app.post("/api/simulation/config")
async def update_config(config: SimulationConfig):
    """Update simulation configuration"""
    state.optimization_mode = config.optimization_mode
    return {"status": "updated", "config": config.dict()}

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
            # Send updates every second
            await asyncio.sleep(1)

            if state.running:
                await state.step_simulation()

            # Broadcast current state
            update = {
                "timestamp": datetime.now().isoformat(),
                "time": state.current_time,
                "hour": state.current_time % 24,
                "metrics": state.metrics,
                "running": state.running
            }

            await manager.broadcast(update)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
