"""
VQE-based Workload Scheduling
Optimizes when to run workloads based on energy cost and carbon intensity
"""

import numpy as np
from typing import Dict, Optional
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP


class WorkloadScheduler:
    """VQE-based workload scheduling optimizer"""

    def __init__(self, use_quantum: bool = True, max_iter: int = 100):
        """
        Initialize workload scheduler.

        Args:
            use_quantum: If True, use VQE. If False, use classical.
            max_iter: Maximum iterations for optimizer
        """
        self.use_quantum = use_quantum
        self.max_iter = max_iter

    def create_hamiltonian(
        self,
        workloads: np.ndarray,
        energy_costs: np.ndarray,
        carbon_intensity: np.ndarray,
        energy_weight: float = 1.0,
        carbon_weight: float = 0.5
    ) -> SparsePauliOp:
        """
        Create Hamiltonian for scheduling problem.

        Args:
            workloads: Workload demand per timeslot
            energy_costs: Electricity price per timeslot
            carbon_intensity: Carbon intensity per timeslot
            energy_weight: Weight for energy cost
            carbon_weight: Weight for carbon emissions

        Returns:
            SparsePauliOp for optimization
        """
        num_timeslots = len(workloads)
        pauli_list = []

        # Energy cost objective
        for t in range(num_timeslots):
            cost = energy_weight * workloads[t] * energy_costs[t]
            pauli_str = 'I' * t + 'Z' + 'I' * (num_timeslots - t - 1)
            pauli_list.append((pauli_str, cost))

        # Carbon emission objective
        for t in range(num_timeslots):
            carbon_cost = carbon_weight * workloads[t] * carbon_intensity[t] * 0.001
            pauli_str = 'I' * t + 'Z' + 'I' * (num_timeslots - t - 1)
            pauli_list.append((pauli_str, carbon_cost))

        return SparsePauliOp.from_list(pauli_list)

    def solve_vqe(
        self,
        workloads: np.ndarray,
        energy_costs: np.ndarray,
        carbon_intensity: np.ndarray,
        reps: int = 2
    ) -> Dict:
        """
        Solve using VQE.

        Args:
            workloads: Workload per timeslot
            energy_costs: Energy cost per timeslot
            carbon_intensity: Carbon intensity per timeslot
            reps: Number of ansatz repetitions

        Returns:
            Dictionary with results
        """
        num_timeslots = len(workloads)

        hamiltonian = self.create_hamiltonian(
            workloads, energy_costs, carbon_intensity
        )

        ansatz = RealAmplitudes(num_timeslots, reps=reps)
        optimizer = SLSQP(maxiter=self.max_iter)
        estimator = StatevectorEstimator()

        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
        result = vqe.compute_minimum_eigenvalue(hamiltonian)

        return {
            'optimal_value': result.optimal_value,
            'optimal_parameters': result.optimal_parameters,
            'result': result
        }

    def solve_classical(
        self,
        workloads: np.ndarray,
        energy_costs: np.ndarray,
        carbon_intensity: np.ndarray,
        energy_weight: float = 1.0,
        carbon_weight: float = 0.5
    ) -> Dict:
        """
        Solve using classical cost-based scheduling.

        Args:
            workloads: Workload per timeslot
            energy_costs: Energy cost per timeslot
            carbon_intensity: Carbon intensity per timeslot
            energy_weight: Weight for energy cost
            carbon_weight: Weight for carbon

        Returns:
            Dictionary with schedule and costs
        """
        # Calculate total cost per timeslot
        total_costs = (
            energy_weight * workloads * energy_costs +
            carbon_weight * workloads * carbon_intensity * 0.001
        )

        # Priority order (lower cost = higher priority)
        schedule_priority = np.argsort(total_costs)

        # Find best timeslots
        best_timeslots = schedule_priority[:len(schedule_priority)//3]  # Top 33%
        medium_timeslots = schedule_priority[len(schedule_priority)//3:2*len(schedule_priority)//3]
        worst_timeslots = schedule_priority[2*len(schedule_priority)//3:]

        return {
            'total_costs': total_costs,
            'schedule_priority': schedule_priority,
            'best_timeslots': best_timeslots,
            'medium_timeslots': medium_timeslots,
            'worst_timeslots': worst_timeslots,
            'min_cost': np.sum(total_costs),
            'best_hour': schedule_priority[0],
            'worst_hour': schedule_priority[-1]
        }

    def optimize(
        self,
        workloads: np.ndarray,
        energy_costs: np.ndarray,
        carbon_intensity: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Optimize workload scheduling.

        Args:
            workloads: Workload demand per timeslot
            energy_costs: Electricity price per timeslot ($/kWh)
            carbon_intensity: Carbon intensity per timeslot (g CO2/kWh)
            **kwargs: Additional arguments

        Returns:
            Dictionary with schedule, costs, and recommendations
        """
        energy_weight = kwargs.get('energy_weight', 1.0)
        carbon_weight = kwargs.get('carbon_weight', 0.5)

        try:
            if self.use_quantum and len(workloads) <= 12:
                # Use VQE for small problems
                vqe_result = self.solve_vqe(
                    workloads, energy_costs, carbon_intensity,
                    reps=kwargs.get('reps', 2)
                )
                classical_result = self.solve_classical(
                    workloads, energy_costs, carbon_intensity,
                    energy_weight, carbon_weight
                )
                method = 'VQE'
                optimal_value = vqe_result['optimal_value']
            else:
                classical_result = self.solve_classical(
                    workloads, energy_costs, carbon_intensity,
                    energy_weight, carbon_weight
                )
                vqe_result = None
                method = 'Classical'
                optimal_value = classical_result['min_cost']

        except Exception as e:
            classical_result = self.solve_classical(
                workloads, energy_costs, carbon_intensity,
                energy_weight, carbon_weight
            )
            vqe_result = None
            method = f'Classical (fallback: {str(e)[:50]})'
            optimal_value = classical_result['min_cost']

        # Generate recommendations
        recommendations = self._generate_recommendations(
            classical_result, workloads, energy_costs, carbon_intensity
        )

        return {
            'schedule': classical_result,
            'vqe_result': vqe_result,
            'optimal_value': optimal_value,
            'method': method,
            'recommendations': recommendations
        }

    def _generate_recommendations(
        self,
        schedule: Dict,
        workloads: np.ndarray,
        energy_costs: np.ndarray,
        carbon_intensity: np.ndarray
    ) -> Dict:
        """Generate scheduling recommendations"""
        best_hours = schedule['best_timeslots']
        worst_hours = schedule['worst_timeslots']

        # Calculate potential savings
        best_avg_cost = np.mean(schedule['total_costs'][best_hours])
        worst_avg_cost = np.mean(schedule['total_costs'][worst_hours])
        potential_savings = (worst_avg_cost - best_avg_cost) / worst_avg_cost

        # Identify renewable-heavy periods (low carbon)
        renewable_periods = np.where(
            carbon_intensity < np.percentile(carbon_intensity, 33)
        )[0]

        # Identify cheap energy periods
        cheap_periods = np.where(
            energy_costs < np.percentile(energy_costs, 33)
        )[0]

        return {
            'best_hours': best_hours.tolist(),
            'worst_hours': worst_hours.tolist(),
            'potential_savings_pct': potential_savings * 100,
            'renewable_periods': renewable_periods.tolist(),
            'cheap_periods': cheap_periods.tolist(),
            'avg_cost_best': best_avg_cost,
            'avg_cost_worst': worst_avg_cost
        }
