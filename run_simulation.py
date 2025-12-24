#!/usr/bin/env python3
"""
Simple script to run the simulation and watch it progress
Works without WebSocket by manually stepping the simulation
"""

import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def reset_simulation():
    """Reset the simulation to initial state"""
    response = requests.post(f"{BASE_URL}/api/simulation/reset")
    print("✅ Simulation reset")
    return response.json()

def start_simulation():
    """Start the simulation"""
    response = requests.post(f"{BASE_URL}/api/simulation/start")
    print("▶️  Simulation started\n")
    return response.json()

def step_simulation():
    """Advance simulation by one step"""
    response = requests.post(f"{BASE_URL}/api/simulation/step")
    return response.json()

def get_status():
    """Get current status"""
    response = requests.get(f"{BASE_URL}/api/status")
    return response.json()

def run_simulation(steps=24, delay=0.5):
    """Run simulation for specified number of steps"""
    # Reset and start
    reset_simulation()
    start_simulation()

    print("┌─────┬──────┬────────────┬──────────┬──────────┬─────────────┬─────┐")
    print("│ Step│ Hour │   Energy   │   Cost   │  Carbon  │ Utilization │ Opts│")
    print("│     │      │    (kWh)   │   ($)    │ (kg CO2) │     (%)     │     │")
    print("├─────┼──────┼────────────┼──────────┼──────────┼─────────────┼─────┤")

    for i in range(1, steps + 1):
        # Step the simulation
        result = step_simulation()

        if result.get('status') == 'stepped':
            metrics = result['metrics']
            hour = result['hour']

            # Format output
            print(f"│ {i:3d} │ {hour:02d}:00│ {metrics['total_energy']:8.2f}   │"
                  f" ${metrics['cost_per_hour']:7.2f} │"
                  f" {metrics['carbon_emissions']/1000:7.2f}  │"
                  f" {metrics['server_utilization']:10.1f}  │"
                  f" {metrics['optimization_count']:3d} │")

            # Highlight optimization steps
            if i % 5 == 0:
                print("│     │ ⚡ Automatic optimization triggered                               │")

        time.sleep(delay)

    print("└─────┴──────┴────────────┴──────────┴──────────┴─────────────┴─────┘")

    # Final status
    status = get_status()
    print(f"\n📊 Final Status:")
    print(f"   • Total time: {status['current_time']} hours")
    print(f"   • Active servers: {status['num_active_servers']}/{status['num_servers']}")
    print(f"   • Tasks: {status['num_tasks']}")
    print(f"   • Optimizations: {status['metrics']['optimization_count']}")
    print(f"   • Anomalies: {status['metrics']['anomalies_detected']}")

def watch_simulation(interval=1.0):
    """Continuously step and watch the simulation"""
    print("👁️  Watching simulation (Ctrl+C to stop)...\n")

    try:
        step_count = 0
        while True:
            result = step_simulation()

            if result.get('status') == 'stepped':
                step_count += 1
                metrics = result['metrics']
                hour = result['hour']

                # Clear line and print update
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Step {step_count:3d} | Hour {hour:02d}:00 | "
                      f"Energy: {metrics['total_energy']:6.2f}kW | "
                      f"Cost: ${metrics['cost_per_hour']:6.2f} | "
                      f"Opts: {metrics['optimization_count']:2d}",
                      end='', flush=True)

                if step_count % 5 == 0:
                    print(" ⚡", end='')

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped watching")

if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("  Quantum Data Center Simulation Runner")
    print("=" * 80)
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] == "watch":
            # Continuous watch mode
            start_simulation()
            watch_simulation()
        elif sys.argv[1].isdigit():
            # Run for N steps
            steps = int(sys.argv[1])
            run_simulation(steps=steps)
        else:
            print("Usage:")
            print("  python run_simulation.py          # Run 24 steps (1 day)")
            print("  python run_simulation.py 48       # Run 48 steps (2 days)")
            print("  python run_simulation.py watch    # Watch continuously")
    else:
        # Default: run 24 steps
        run_simulation(steps=24, delay=0.3)

    print("\n✅ Simulation complete!")
