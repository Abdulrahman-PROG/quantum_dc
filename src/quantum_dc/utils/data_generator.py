"""
Data Generator for Data Center Quantum Optimization
Generates realistic synthetic data for testing quantum algorithms
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class DataCenterConfig:
    """Configuration for data center simulation"""
    num_servers: int = 10
    num_tasks: int = 20
    num_zones: int = 5
    num_timeslots: int = 24
    outdoor_temp_min: float = 15.0  # °C
    outdoor_temp_max: float = 35.0  # °C
    base_carbon_intensity: float = 400.0  # g CO2/kWh
    base_electricity_price: float = 0.15  # $/kWh


class DataCenterDataGenerator:
    """Generate realistic data center operation data"""

    def __init__(self, config: DataCenterConfig = None):
        self.config = config or DataCenterConfig()
        np.random.seed(42)  # For reproducibility

    def generate_server_data(self) -> Dict[str, np.ndarray]:
        """
        Generate server specifications.

        Returns:
            Dictionary with server capacities, power consumption, efficiency
        """
        # Server capacities (CPU cores): Mix of small, medium, large servers
        capacities = np.random.choice(
            [100, 200, 400, 800],  # CPU cores
            size=self.config.num_servers,
            p=[0.3, 0.4, 0.2, 0.1]  # More medium servers
        )

        # Base power consumption (watts) - scales with capacity
        base_power = capacities * 1.5 + np.random.normal(0, 20, self.config.num_servers)
        base_power = np.maximum(base_power, 50)  # Minimum 50W

        # Power efficiency (PUE - Power Usage Effectiveness)
        # 1.0 is perfect, typical is 1.2-2.0
        pue = np.random.uniform(1.2, 1.8, self.config.num_servers)

        # Server utilization efficiency (0-1)
        efficiency = np.random.uniform(0.7, 0.95, self.config.num_servers)

        return {
            'capacities': capacities.astype(float),
            'base_power': base_power,
            'pue': pue,
            'efficiency': efficiency
        }

    def generate_task_data(self) -> Dict[str, np.ndarray]:
        """
        Generate task/workload specifications.

        Returns:
            Dictionary with task CPU requirements, memory, priority
        """
        # Task CPU loads (cores required)
        # Mix of small (web requests), medium (batch jobs), large (ML training)
        cpu_loads = np.random.choice(
            [10, 25, 50, 100, 200],
            size=self.config.num_tasks,
            p=[0.4, 0.3, 0.2, 0.08, 0.02]  # Most tasks are small
        )

        # Memory requirements (GB)
        memory_loads = cpu_loads * np.random.uniform(2, 8, self.config.num_tasks)

        # Task priority (1-10, higher = more important)
        priorities = np.random.randint(1, 11, self.config.num_tasks)

        # Task duration (hours)
        durations = np.random.exponential(2.0, self.config.num_tasks)
        durations = np.clip(durations, 0.1, 12.0)

        # Task flexibility (can it be delayed?)
        flexibility = np.random.uniform(0, 1, self.config.num_tasks)

        return {
            'cpu_loads': cpu_loads.astype(float),
            'memory_loads': memory_loads,
            'priorities': priorities,
            'durations': durations,
            'flexibility': flexibility
        }

    def generate_timeseries_data(self) -> Dict[str, np.ndarray]:
        """
        Generate time-varying data (energy prices, carbon, temperature).

        Returns:
            Dictionary with hourly data for 24 hours
        """
        hours = np.arange(self.config.num_timeslots)

        # Outdoor temperature (°C) - follows daily pattern
        # Peak at 2pm, minimum at 4am
        temp_amplitude = (self.config.outdoor_temp_max - self.config.outdoor_temp_min) / 2
        temp_mean = (self.config.outdoor_temp_max + self.config.outdoor_temp_min) / 2
        outdoor_temp = temp_mean + temp_amplitude * np.sin(
            2 * np.pi * (hours - 4) / 24
        )
        # Add some noise
        outdoor_temp += np.random.normal(0, 2, self.config.num_timeslots)

        # Electricity price ($/kWh) - higher during peak hours (6pm-10pm)
        # Off-peak: 0.10, peak: 0.25
        base_price = self.config.base_electricity_price
        price_variation = 0.10 * (
            np.sin(2 * np.pi * (hours - 6) / 24) * 0.5 + 0.5
        )
        electricity_price = base_price + price_variation
        # Peak hours (18-22)
        peak_mask = (hours >= 18) & (hours <= 22)
        electricity_price[peak_mask] *= 1.5

        # Carbon intensity (g CO2/kWh) - lower when renewables are abundant
        # Solar peak at noon, wind variable
        solar_pattern = np.maximum(
            0, np.sin(2 * np.pi * (hours - 6) / 24)
        )
        wind_pattern = 0.3 + 0.2 * np.sin(2 * np.pi * hours / 24 + np.pi/4)

        renewable_fraction = 0.3 * solar_pattern + 0.2 * wind_pattern
        carbon_intensity = self.config.base_carbon_intensity * (1 - renewable_fraction)
        carbon_intensity += np.random.normal(0, 20, self.config.num_timeslots)
        carbon_intensity = np.maximum(carbon_intensity, 100)  # Min 100 g/kWh

        # Workload demand (normalized 0-1) - peaks during business hours
        workload_demand = 0.3 + 0.7 * (
            0.5 + 0.5 * np.sin(2 * np.pi * (hours - 9) / 12)
        )
        # Weekend vs weekday (assume weekday)
        workload_demand = np.clip(workload_demand, 0.2, 1.0)

        return {
            'outdoor_temp': outdoor_temp,
            'electricity_price': electricity_price,
            'carbon_intensity': carbon_intensity,
            'workload_demand': workload_demand,
            'hours': hours
        }

    def generate_cooling_data(self) -> Dict[str, np.ndarray]:
        """
        Generate cooling zone data.

        Returns:
            Dictionary with heat loads, current temps, setpoints
        """
        # Heat load per zone (kW) - varies by zone
        heat_loads = np.random.uniform(30, 80, self.config.num_zones)

        # Current temperature per zone (°C)
        current_temps = np.random.uniform(20, 25, self.config.num_zones)

        # Current setpoints (°C)
        current_setpoints = np.random.uniform(18, 24, self.config.num_zones)

        # Cooling efficiency (COP - Coefficient of Performance)
        # Typical range 2.5-4.0 (higher is better)
        cooling_cop = np.random.uniform(2.5, 4.0, self.config.num_zones)

        return {
            'heat_loads': heat_loads,
            'current_temps': current_temps,
            'current_setpoints': current_setpoints,
            'cooling_cop': cooling_cop
        }

    def generate_sensor_data(self, num_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate sensor data for anomaly detection and predictive maintenance.

        Args:
            num_samples: Number of sensor readings

        Returns:
            Dictionary with sensor readings and failure indicators
        """
        # Temperature sensors (°C)
        temp_normal = np.random.normal(23, 3, num_samples)

        # Vibration sensors (mm/s)
        vibration_normal = np.random.exponential(2, num_samples)

        # Power consumption (kW)
        power_normal = np.random.normal(100, 15, num_samples)

        # Current (Amperes)
        current_normal = np.random.normal(45, 8, num_samples)

        # Voltage (Volts)
        voltage_normal = np.random.normal(220, 5, num_samples)

        # Humidity (%)
        humidity_normal = np.random.normal(50, 10, num_samples)

        # Equipment age (years)
        age = np.random.uniform(0, 10, num_samples)

        # Operating cycles (thousands)
        cycles = np.random.uniform(0, 100, num_samples)

        # Create failure probability based on conditions
        failure_prob = (
            0.01 * (temp_normal - 23) +  # Temp deviation
            0.05 * vibration_normal +     # Vibration
            0.02 * age +                  # Age
            0.001 * cycles                # Wear
        )
        failure_prob = 1 / (1 + np.exp(-failure_prob))  # Sigmoid

        # Binary failure indicator
        failures = (failure_prob > 0.5).astype(int)

        # Add some anomalies (10% of data)
        anomaly_indices = np.random.choice(
            num_samples, size=int(0.1 * num_samples), replace=False
        )
        temp_normal[anomaly_indices] += np.random.uniform(10, 20, len(anomaly_indices))
        vibration_normal[anomaly_indices] *= np.random.uniform(2, 4, len(anomaly_indices))
        power_normal[anomaly_indices] *= np.random.uniform(1.5, 2.5, len(anomaly_indices))

        return {
            'temperature': temp_normal,
            'vibration': vibration_normal,
            'power': power_normal,
            'current': current_normal,
            'voltage': voltage_normal,
            'humidity': humidity_normal,
            'age': age,
            'cycles': cycles,
            'failure_prob': failure_prob,
            'failures': failures,
            'anomaly_indices': anomaly_indices
        }

    def generate_battery_data(self) -> Dict[str, float]:
        """
        Generate battery system specifications.

        Returns:
            Dictionary with battery capacity, efficiency, etc.
        """
        return {
            'capacity': 1000.0,  # kWh
            'max_charge_rate': 200.0,  # kW
            'max_discharge_rate': 200.0,  # kW
            'charge_efficiency': 0.95,  # 95% round-trip efficiency
            'discharge_efficiency': 0.95,
            'min_soc': 0.1,  # Minimum 10% state of charge
            'max_soc': 0.9,  # Maximum 90% state of charge
            'initial_soc': 0.5  # Start at 50%
        }

    def generate_workload_features(self, num_samples: int = 200) -> Dict[str, np.ndarray]:
        """
        Generate workload characteristics for clustering.

        Args:
            num_samples: Number of workload samples

        Returns:
            Dictionary with workload features
        """
        # Create 3 types of workloads
        n_per_type = num_samples // 3

        # Type 1: CPU-intensive (e.g., computation, ML training)
        cpu_intensive = np.random.randn(n_per_type, 4) * 0.15 + np.array([0.9, 0.3, 0.2, 0.6])

        # Type 2: Memory-intensive (e.g., databases, caching)
        mem_intensive = np.random.randn(n_per_type, 4) * 0.15 + np.array([0.4, 0.9, 0.3, 0.7])

        # Type 3: Network-intensive (e.g., web servers, streaming)
        net_intensive = np.random.randn(n_per_type, 4) * 0.15 + np.array([0.3, 0.4, 0.9, 0.5])

        # Combine
        features = np.vstack([cpu_intensive, mem_intensive, net_intensive])
        features = np.clip(features, 0, 1)  # Ensure valid range

        # True labels for validation
        labels = np.concatenate([
            np.zeros(n_per_type),
            np.ones(n_per_type),
            np.full(n_per_type, 2)
        ]).astype(int)

        return {
            'features': features,  # [CPU usage, Memory usage, Network I/O, Duration]
            'labels': labels,
            'feature_names': ['CPU', 'Memory', 'Network', 'Duration']
        }

    def generate_complete_scenario(self) -> Dict[str, Dict]:
        """
        Generate a complete data center scenario with all components.

        Returns:
            Dictionary containing all generated data
        """
        return {
            'servers': self.generate_server_data(),
            'tasks': self.generate_task_data(),
            'timeseries': self.generate_timeseries_data(),
            'cooling': self.generate_cooling_data(),
            'sensors': self.generate_sensor_data(),
            'battery': self.generate_battery_data(),
            'workloads': self.generate_workload_features(),
            'config': self.config
        }


def create_sample_dataset(scenario: str = 'balanced') -> Dict:
    """
    Create pre-configured sample datasets for different scenarios.

    Args:
        scenario: One of 'small', 'balanced', 'large', 'high_load'

    Returns:
        Complete dataset for the scenario
    """
    configs = {
        'small': DataCenterConfig(
            num_servers=5, num_tasks=10, num_zones=3, num_timeslots=12
        ),
        'balanced': DataCenterConfig(
            num_servers=10, num_tasks=20, num_zones=5, num_timeslots=24
        ),
        'large': DataCenterConfig(
            num_servers=20, num_tasks=50, num_zones=10, num_timeslots=24
        ),
        'high_load': DataCenterConfig(
            num_servers=10, num_tasks=30, num_zones=5, num_timeslots=24
        )
    }

    config = configs.get(scenario, configs['balanced'])
    generator = DataCenterDataGenerator(config)
    return generator.generate_complete_scenario()
