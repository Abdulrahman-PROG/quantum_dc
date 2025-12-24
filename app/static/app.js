// Quantum Data Center Control - Frontend JavaScript

// Global state
let ws = null;
let charts = {};
let previousMetrics = {};
let eventLogMaxEntries = 50;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeWebSocket();
    initializeControls();
    initializeCharts();
    loadInitialData();
});

// WebSocket Connection
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        updateConnectionStatus(true);
        addEvent('Connected to data center', 'success');
    };

    ws.onclose = () => {
        updateConnectionStatus(false);
        addEvent('Disconnected from data center', 'error');
        // Attempt to reconnect after 3 seconds
        setTimeout(initializeWebSocket, 3000);
    };

    ws.onerror = (error) => {
        addEvent('Connection error', 'error');
        console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleRealtimeUpdate(data);
    };
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (connected) {
        statusElement.textContent = 'Connected';
        statusElement.className = 'badge badge-connected';
    } else {
        statusElement.textContent = 'Disconnected';
        statusElement.className = 'badge badge-disconnected';
    }
}

// Control Handlers
function initializeControls() {
    document.getElementById('btn-start').addEventListener('click', startSimulation);
    document.getElementById('btn-stop').addEventListener('click', stopSimulation);
    document.getElementById('btn-reset').addEventListener('click', resetSimulation);
    document.getElementById('btn-optimize').addEventListener('click', runOptimization);
    document.getElementById('optimization-mode').addEventListener('change', updateOptimizationMode);
}

async function startSimulation() {
    try {
        const response = await fetch('/api/simulation/start', { method: 'POST' });
        const data = await response.json();
        addEvent('Simulation started', 'success');
        updateSimulationStatus(true);
    } catch (error) {
        addEvent('Failed to start simulation', 'error');
        console.error(error);
    }
}

async function stopSimulation() {
    try {
        const response = await fetch('/api/simulation/stop', { method: 'POST' });
        const data = await response.json();
        addEvent('Simulation stopped', 'warning');
        updateSimulationStatus(false);
    } catch (error) {
        addEvent('Failed to stop simulation', 'error');
        console.error(error);
    }
}

async function resetSimulation() {
    try {
        const response = await fetch('/api/simulation/reset', { method: 'POST' });
        const data = await response.json();
        addEvent('Simulation reset', 'info');
        updateSimulationStatus(false);
        loadInitialData();
    } catch (error) {
        addEvent('Failed to reset simulation', 'error');
        console.error(error);
    }
}

async function runOptimization() {
    const mode = document.getElementById('optimization-mode').value;

    if (mode === 'none') {
        addEvent('Optimization disabled', 'warning');
        return;
    }

    try {
        addEvent(`Running ${mode.toUpperCase()} optimization...`, 'info');

        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ method: mode === 'auto' ? 'qaoa' : mode })
        });

        const result = await response.json();

        if (result.success) {
            addEvent(`${result.method.toUpperCase()} optimization complete`, 'success');
            loadServersAndTasks();
        } else {
            addEvent('Optimization failed', 'error');
        }
    } catch (error) {
        addEvent('Optimization error', 'error');
        console.error(error);
    }
}

async function updateOptimizationMode() {
    const mode = document.getElementById('optimization-mode').value;

    try {
        await fetch('/api/simulation/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                optimization_mode: mode,
                num_servers: 10,
                num_tasks: 15
            })
        });

        addEvent(`Optimization mode: ${mode.toUpperCase()}`, 'info');
    } catch (error) {
        console.error(error);
    }
}

function updateSimulationStatus(running) {
    const statusElement = document.getElementById('simulation-status');
    if (running) {
        statusElement.textContent = 'Running';
        statusElement.className = 'badge badge-running';
    } else {
        statusElement.textContent = 'Stopped';
        statusElement.className = 'badge badge-stopped';
    }
}

// Load Initial Data
async function loadInitialData() {
    await loadStatus();
    await loadServersAndTasks();
    await loadTimeseries();
}

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        updateMetrics(data.metrics);
        updateTime(data.time, data.hour);
        updateSimulationStatus(data.running);
    } catch (error) {
        console.error('Failed to load status:', error);
    }
}

async function loadServersAndTasks() {
    try {
        const [serversResponse, tasksResponse] = await Promise.all([
            fetch('/api/servers'),
            fetch('/api/tasks')
        ]);

        const servers = await serversResponse.json();
        const tasks = await tasksResponse.json();

        renderServerRack(servers);
        renderTaskQueue(tasks);
        updateLoadChart(servers);
    } catch (error) {
        console.error('Failed to load servers/tasks:', error);
    }
}

async function loadTimeseries() {
    try {
        const response = await fetch('/api/timeseries');
        const data = await response.json();
        updateEnergyChart(data);
        updateEnvironmentChart(data);
    } catch (error) {
        console.error('Failed to load timeseries:', error);
    }
}

// Realtime Updates
function handleRealtimeUpdate(data) {
    updateMetrics(data.metrics);
    updateTime(data.time, data.hour);
    updateSimulationStatus(data.running);

    // Reload visualizations periodically
    if (data.time % 5 === 0) {
        loadServersAndTasks();
    }
}

// Update Metrics with Trends
function updateMetrics(metrics) {
    updateMetric('energy', metrics.total_energy.toFixed(2), 'kWh');
    updateMetric('cost', metrics.cost_per_hour.toFixed(2), '$', true);
    updateMetric('temp', metrics.average_temp.toFixed(1), '°C');
    updateMetric('carbon', metrics.carbon_emissions.toFixed(0), 'g CO₂');
    updateMetric('utilization', metrics.server_utilization.toFixed(1), '%');

    document.getElementById('metric-optimizations').textContent = metrics.optimization_count;

    // Calculate and display trends
    for (const key in metrics) {
        if (previousMetrics[key] !== undefined) {
            const diff = metrics[key] - previousMetrics[key];
            const trendElement = document.getElementById(`trend-${getTrendKey(key)}`);

            if (trendElement && diff !== 0) {
                const arrow = diff > 0 ? '↑' : '↓';
                const color = shouldBeIncreasing(key) ?
                    (diff > 0 ? 'var(--success-color)' : 'var(--danger-color)') :
                    (diff < 0 ? 'var(--success-color)' : 'var(--danger-color)');

                trendElement.textContent = `${arrow} ${Math.abs(diff).toFixed(2)}`;
                trendElement.style.color = color;
            }
        }
    }

    previousMetrics = { ...metrics };
}

function updateMetric(key, value, unit, isCurrency = false) {
    const element = document.getElementById(`metric-${key}`);
    if (element) {
        element.textContent = isCurrency ? `$${value}` : `${value} ${unit}`;
    }
}

function getTrendKey(metricKey) {
    const mapping = {
        'total_energy': 'energy',
        'cost_per_hour': 'cost',
        'average_temp': 'temp',
        'carbon_emissions': 'carbon',
        'server_utilization': 'utilization',
        'optimization_count': 'optimizations'
    };
    return mapping[metricKey] || metricKey;
}

function shouldBeIncreasing(key) {
    return ['server_utilization', 'optimization_count'].includes(key);
}

function updateTime(time, hour) {
    document.getElementById('current-time').textContent = time;
    document.getElementById('current-hour').textContent = hour.toString().padStart(2, '0');
}

// Render Server Rack
function renderServerRack(servers) {
    const container = document.getElementById('server-rack');
    container.innerHTML = '';

    servers.forEach(server => {
        const serverDiv = document.createElement('div');
        serverDiv.className = 'server';

        if (server.utilization > 80) {
            serverDiv.classList.add('critical');
        } else if (server.utilization > 60) {
            serverDiv.classList.add('warning');
        } else if (server.status === 'active') {
            serverDiv.classList.add('active');
        }

        serverDiv.innerHTML = `
            <div class="server-id">S${server.id}</div>
            <div class="server-load">${server.utilization.toFixed(0)}%</div>
            <div class="server-bar">
                <div class="server-bar-fill" style="width: ${server.utilization}%"></div>
            </div>
        `;

        serverDiv.title = `Server ${server.id}\nCapacity: ${server.capacity} cores\nLoad: ${server.current_load.toFixed(0)}/${server.capacity}\nUtilization: ${server.utilization.toFixed(1)}%`;

        container.appendChild(serverDiv);
    });
}

// Render Task Queue
function renderTaskQueue(tasks) {
    const container = document.getElementById('task-queue');
    container.innerHTML = '';

    tasks.slice(0, 20).forEach(task => {
        const taskDiv = document.createElement('div');
        taskDiv.className = 'task-item';

        taskDiv.innerHTML = `
            <div class="task-info">
                <div class="task-id">Task #${task.id}</div>
                <div class="task-details">${task.cpu_cores} cores, ${task.memory_gb} GB</div>
            </div>
            <div class="task-server">S${task.assigned_server}</div>
        `;

        container.appendChild(taskDiv);
    });

    if (tasks.length > 20) {
        const moreDiv = document.createElement('div');
        moreDiv.className = 'task-item';
        moreDiv.innerHTML = `<div class="task-info">+ ${tasks.length - 20} more tasks...</div>`;
        container.appendChild(moreDiv);
    }
}

// Initialize Charts
function initializeCharts() {
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                labels: { color: '#f1f5f9' }
            }
        },
        scales: {
            x: {
                ticks: { color: '#94a3b8' },
                grid: { color: '#334155' }
            },
            y: {
                ticks: { color: '#94a3b8' },
                grid: { color: '#334155' }
            }
        }
    };

    // Energy Chart
    const energyCtx = document.getElementById('energy-chart').getContext('2d');
    charts.energy = new Chart(energyCtx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
            datasets: [
                {
                    label: 'Energy Price ($/kWh)',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: chartDefaults
    });

    // Environment Chart
    const envCtx = document.getElementById('environment-chart').getContext('2d');
    charts.environment = new Chart(envCtx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                },
                {
                    label: 'Carbon Intensity (g/kWh)',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }
            ]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    ticks: { color: '#94a3b8' },
                    grid: { display: false }
                }
            }
        }
    });

    // Load Chart
    const loadCtx = document.getElementById('load-chart').getContext('2d');
    charts.load = new Chart(loadCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Server Load (%)',
                data: [],
                backgroundColor: 'rgba(0, 212, 255, 0.6)',
                borderColor: '#00d4ff',
                borderWidth: 1
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

function updateEnergyChart(data) {
    charts.energy.data.datasets[0].data = data.energy_price;
    charts.energy.update();
}

function updateEnvironmentChart(data) {
    charts.environment.data.datasets[0].data = data.temperature;
    charts.environment.data.datasets[1].data = data.carbon_intensity;
    charts.environment.update();
}

function updateLoadChart(servers) {
    charts.load.data.labels = servers.map(s => `S${s.id}`);
    charts.load.data.datasets[0].data = servers.map(s => s.utilization);

    // Color code based on utilization
    charts.load.data.datasets[0].backgroundColor = servers.map(s => {
        if (s.utilization > 80) return 'rgba(239, 68, 68, 0.6)';
        if (s.utilization > 60) return 'rgba(245, 158, 11, 0.6)';
        return 'rgba(0, 212, 255, 0.6)';
    });

    charts.load.update();
}

// Event Log
function addEvent(message, type = 'info') {
    const container = document.getElementById('event-log');
    const eventDiv = document.createElement('div');
    eventDiv.className = `event ${type}`;

    const now = new Date();
    const timeStr = now.toLocaleTimeString();

    eventDiv.innerHTML = `
        <div class="event-time">[${timeStr}]</div>
        <div>${message}</div>
    `;

    container.insertBefore(eventDiv, container.firstChild);

    // Limit event log entries
    while (container.children.length > eventLogMaxEntries) {
        container.removeChild(container.lastChild);
    }
}

// Utility Functions
function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

function getColorForValue(value, thresholds = { low: 50, high: 80 }) {
    if (value < thresholds.low) return 'var(--success-color)';
    if (value < thresholds.high) return 'var(--warning-color)';
    return 'var(--danger-color)';
}
