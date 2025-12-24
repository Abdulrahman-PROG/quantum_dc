"""
Quantum-Inspired Cooling Optimization
Finds optimal temperature setpoints to minimize cooling energy
"""

import numpy as np
from typing import Dict
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver


class CoolingOptimizer:
    """Quantum-inspired cooling setpoint optimizer"""

    def __init__(self, temp_min: float = 18.0, temp_max: float = 24.0, num_levels: int = 7):
        """
        Initialize cooling optimizer.

        Args:
            temp_min: Minimum safe temperature (°C)
            temp_max: Maximum safe temperature (°C)
            num_levels: Number of discrete temperature levels
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.num_levels = num_levels

    def create_qubo(
        self,
        num_zones: int,
        outdoor_temp: float,
        heat_loads: np.ndarray
    ) -> QuadraticProgram:
        """Create QUBO problem for cooling optimization"""
        qp = QuadraticProgram("cooling_optimization")

        # Binary variables for each zone and temperature level
        for zone in range(num_zones):
            for level in range(self.num_levels):
                qp.binary_var(f"z{zone}_t{level}")

        # Objective: minimize cooling power
        linear = {}
        for zone in range(num_zones):
            for level in range(self.num_levels):
                temp_setpoint = self.temp_min + level * (
                    (self.temp_max - self.temp_min) / (self.num_levels - 1)
                )
                # Cooling power inversely proportional to (setpoint - outdoor_temp)
                cooling_power = heat_loads[zone] / max(temp_setpoint - outdoor_temp, 0.1)
                linear[f"z{zone}_t{level}"] = cooling_power

        qp.minimize(linear=linear)

        # Constraint: Each zone has exactly one setpoint
        for zone in range(num_zones):
            zone_vars = {f"z{zone}_t{level}": 1 for level in range(self.num_levels)}
            qp.linear_constraint(zone_vars, '==', 1)

        return qp

    def optimize(
        self,
        outdoor_temp: float,
        heat_loads: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Optimize cooling setpoints.

        Args:
            outdoor_temp: Outdoor temperature (°C)
            heat_loads: Heat load per zone (kW)
            **kwargs: Additional parameters

        Returns:
            Dictionary with optimal setpoints and energy
        """
        num_zones = len(heat_loads)

        try:
            # Create and solve QUBO
            qp = self.create_qubo(num_zones, outdoor_temp, heat_loads)
            optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
            result = optimizer.solve(qp)

            # Extract setpoints
            setpoints = np.zeros(num_zones)
            for zone in range(num_zones):
                for level in range(self.num_levels):
                    var_name = f"z{zone}_t{level}"
                    if var_name in result.variables_dict and result.variables_dict[var_name] == 1:
                        setpoints[zone] = self.temp_min + level * (
                            (self.temp_max - self.temp_min) / (self.num_levels - 1)
                        )
                        break

            total_energy = result.fval
            method = 'QUBO'

        except Exception as e:
            # Fallback: greedy approach (maximize setpoint for efficiency)
            setpoints = np.full(num_zones, self.temp_max)
            total_energy = np.sum(heat_loads / (self.temp_max - outdoor_temp))
            method = f'Greedy (fallback: {str(e)[:50]})'

        # Calculate metrics
        metrics = {
            'avg_setpoint': np.mean(setpoints),
            'setpoint_range': np.max(setpoints) - np.min(setpoints),
            'total_cooling_power': total_energy,
            'power_per_zone': heat_loads / (setpoints - outdoor_temp)
        }

        return {
            'setpoints': setpoints,
            'total_energy': total_energy,
            'method': method,
            'metrics': metrics
        }
