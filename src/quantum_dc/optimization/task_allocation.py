"""
QAOA-based Task Allocation for Data Centers
Assigns tasks to servers to minimize energy consumption
"""

import numpy as np
from typing import Tuple, Dict, Optional
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA


class TaskAllocationOptimizer:
    """Quantum-based task allocation optimizer using QAOA"""

    def __init__(self, use_quantum: bool = True, max_iter: int = 100):
        """
        Initialize task allocation optimizer.

        Args:
            use_quantum: If True, use QAOA. If False, use classical greedy.
            max_iter: Maximum iterations for optimizer
        """
        self.use_quantum = use_quantum
        self.max_iter = max_iter

    def create_hamiltonian(
        self,
        task_loads: np.ndarray,
        server_capacities: np.ndarray,
        energy_weight: float = 1.0,
        constraint_weight: float = 10.0,
        balance_weight: float = 0.5
    ) -> SparsePauliOp:
        """
        Create Hamiltonian for task allocation problem.

        Args:
            task_loads: CPU requirements for each task
            server_capacities: Capacity of each server
            energy_weight: Weight for energy minimization
            constraint_weight: Weight for constraint penalties
            balance_weight: Weight for load balancing

        Returns:
            SparsePauliOp representing the optimization problem
        """
        num_tasks = len(task_loads)
        num_servers = len(server_capacities)
        num_qubits = num_tasks * num_servers

        pauli_list = []

        # Objective 1: Minimize energy consumption
        # Energy cost proportional to load / capacity
        for i in range(num_tasks):
            for j in range(num_servers):
                qubit_idx = i * num_servers + j
                energy_cost = energy_weight * task_loads[i] / server_capacities[j]
                pauli_str = 'I' * qubit_idx + 'Z' + 'I' * (num_qubits - qubit_idx - 1)
                pauli_list.append((pauli_str, energy_cost))

        # Constraint: Each task assigned to exactly one server
        for i in range(num_tasks):
            for j1 in range(num_servers):
                for j2 in range(j1 + 1, num_servers):
                    q1 = i * num_servers + j1
                    q2 = i * num_servers + j2
                    pauli_str = ['I'] * num_qubits
                    pauli_str[q1] = 'Z'
                    pauli_str[q2] = 'Z'
                    pauli_list.append((''.join(pauli_str), constraint_weight))

        # Objective 2: Load balancing
        for j in range(num_servers):
            for i in range(num_tasks):
                qubit_idx = i * num_servers + j
                pauli_str = 'I' * qubit_idx + 'Z' + 'I' * (num_qubits - qubit_idx - 1)
                pauli_list.append((pauli_str, balance_weight))

        return SparsePauliOp.from_list(pauli_list)

    def solve_qaoa(
        self,
        task_loads: np.ndarray,
        server_capacities: np.ndarray,
        reps: int = 1
    ) -> Tuple[np.ndarray, float, Optional[object]]:
        """
        Solve using QAOA.

        Args:
            task_loads: Task CPU requirements
            server_capacities: Server capacities
            reps: QAOA circuit depth

        Returns:
            (allocation_matrix, optimal_energy, result_object)
        """
        num_tasks = len(task_loads)
        num_servers = len(server_capacities)

        hamiltonian = self.create_hamiltonian(task_loads, server_capacities)

        optimizer = COBYLA(maxiter=self.max_iter)
        sampler = StatevectorSampler()
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)

        result = qaoa.compute_minimum_eigenvalue(hamiltonian)

        # Extract allocation from result
        allocation = np.zeros((num_tasks, num_servers), dtype=int)

        # Get best measurement
        if hasattr(result, 'best_measurement'):
            bitstring = result.best_measurement['bitstring']
            for i in range(num_tasks):
                for j in range(num_servers):
                    idx = i * num_servers + j
                    if idx < len(bitstring):
                        allocation[i, j] = int(bitstring[idx])

        return allocation, result.optimal_value, result

    def solve_greedy(
        self,
        task_loads: np.ndarray,
        server_capacities: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Solve using classical greedy algorithm.

        Args:
            task_loads: Task CPU requirements
            server_capacities: Server capacities

        Returns:
            (allocation_matrix, total_energy)
        """
        num_tasks = len(task_loads)
        num_servers = len(server_capacities)

        allocation = np.zeros((num_tasks, num_servers), dtype=int)
        server_loads = np.zeros(num_servers)

        # Sort tasks by load (largest first)
        task_order = np.argsort(task_loads)[::-1]

        for task_idx in task_order:
            load = task_loads[task_idx]

            # Calculate cost for each server (energy + load balance penalty)
            costs = np.zeros(num_servers)
            for j in range(num_servers):
                if server_loads[j] + load <= server_capacities[j]:
                    energy_cost = load / server_capacities[j]
                    balance_penalty = 0.5 * (server_loads[j] / server_capacities[j])
                    costs[j] = energy_cost + balance_penalty
                else:
                    costs[j] = np.inf  # Cannot fit

            # Assign to best server
            best_server = np.argmin(costs)
            if costs[best_server] < np.inf:
                allocation[task_idx, best_server] = 1
                server_loads[best_server] += load

        # Calculate total energy
        total_energy = 0
        for i in range(num_tasks):
            for j in range(num_servers):
                if allocation[i, j] == 1:
                    total_energy += task_loads[i] / server_capacities[j]

        return allocation, total_energy

    def optimize(
        self,
        task_loads: np.ndarray,
        server_capacities: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Optimize task allocation.

        Args:
            task_loads: Array of task CPU requirements
            server_capacities: Array of server capacities
            **kwargs: Additional arguments for QAOA (reps, etc.)

        Returns:
            Dictionary with allocation, energy, method used, and metrics
        """
        try:
            if self.use_quantum and len(task_loads) * len(server_capacities) <= 12:
                # Use QAOA for small problems
                allocation, energy, result = self.solve_qaoa(
                    task_loads, server_capacities, **kwargs
                )
                method = 'QAOA'
            else:
                # Use greedy for large problems or if quantum disabled
                allocation, energy = self.solve_greedy(task_loads, server_capacities)
                result = None
                method = 'Greedy'

        except Exception as e:
            # Fallback to greedy on error
            allocation, energy = self.solve_greedy(task_loads, server_capacities)
            result = None
            method = f'Greedy (fallback: {str(e)[:50]})'

        # Calculate metrics
        metrics = self._calculate_metrics(allocation, task_loads, server_capacities)

        return {
            'allocation': allocation,
            'energy': energy,
            'method': method,
            'metrics': metrics,
            'result': result
        }

    def _calculate_metrics(
        self,
        allocation: np.ndarray,
        task_loads: np.ndarray,
        server_capacities: np.ndarray
    ) -> Dict:
        """Calculate performance metrics"""
        num_tasks, num_servers = allocation.shape

        # Server utilization
        server_loads = np.zeros(num_servers)
        for i in range(num_tasks):
            for j in range(num_servers):
                if allocation[i, j] == 1:
                    server_loads[j] += task_loads[i]

        utilization = server_loads / server_capacities
        active_servers = np.sum(server_loads > 0)

        # Load balance (std of utilization)
        load_balance = 1.0 - np.std(utilization)

        # Task assignment success
        assigned_tasks = np.sum(np.any(allocation, axis=1))

        return {
            'utilization': utilization,
            'avg_utilization': np.mean(utilization),
            'max_utilization': np.max(utilization),
            'load_balance': load_balance,
            'active_servers': active_servers,
            'assigned_tasks': assigned_tasks,
            'total_tasks': num_tasks
        }
