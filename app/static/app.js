// Quantum Data Center Control - Frontend JavaScript

// Global state
let ws = null;
let charts = {};
let previousMetrics = {};
let eventLogMaxEntries = 50;
let wsReconnectDelay = 1000; // Start at 1s, exponential backoff
const WS_MAX_RECONNECT_DELAY = 30000;

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
        wsReconnectDelay = 1000; // Reset backoff on successful connection
    };

    ws.onclose = () => {
        updateConnectionStatus(false);
        addEvent('Disconnected from data center', 'error');
        // Reconnect with exponential backoff
        setTimeout(initializeWebSocket, wsReconnectDelay);
        wsReconnectDelay = Math.min(wsReconnectDelay * 2, WS_MAX_RECONNECT_DELAY);
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
    document.getElementById('btn-benchmark').addEventListener('click', runBenchmark);
    document.getElementById('btn-carbon').addEventListener('click', runCarbonOptimizer);
    document.getElementById('btn-history').addEventListener('click', loadHistory);
    document.getElementById('btn-forecast').addEventListener('click', loadForecast);
}

async function startSimulation() {
    try {
        await fetch('/api/simulation/start', { method: 'POST' });
        addEvent('Simulation started', 'success');
        updateSimulationStatus(true);
    } catch (error) {
        addEvent('Failed to start simulation', 'error');
        console.error(error);
    }
}

async function stopSimulation() {
    try {
        await fetch('/api/simulation/stop', { method: 'POST' });
        addEvent('Simulation stopped', 'warning');
        updateSimulationStatus(false);
    } catch (error) {
        addEvent('Failed to stop simulation', 'error');
        console.error(error);
    }
}

async function resetSimulation() {
    try {
        await fetch('/api/simulation/reset', { method: 'POST' });
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

// Load Initial Data — run independent fetches in parallel
async function loadInitialData() {
    await Promise.all([loadStatus(), loadServersAndTasks(), loadTimeseries()]);
}

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        updateMetrics(data.metrics);
        updateTime(data.current_time, data.hour);
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
    updateTime(data.current_time, data.hour);
    updateSimulationStatus(data.running);

    // Reload visualizations periodically
    if (data.current_time % 5 === 0) {
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
                <div class="task-details">${task.cpu_cores} cores, ${Number(task.memory_gb).toFixed(1)} GB</div>
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

// LSTM Energy Forecast
async function loadForecast() {
    document.getElementById('btn-forecast').disabled = true;
    try {
        const response = await fetch('/api/forecast');
        if (!response.ok) {
            const err = await response.json();
            addEvent('Forecast error: ' + err.detail, 'error');
            return;
        }
        const data = await response.json();

        document.getElementById('forecast-model-badge').textContent =
            `Model: ${data.model}`;

        if (charts.forecast) charts.forecast.destroy();

        const ctx = document.getElementById('forecast-chart').getContext('2d');
        charts.forecast = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.hours.map(h => `${h}:00`),
                datasets: [
                    {
                        label: 'Predicted Energy (kW)',
                        data: data.predicted,
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0,212,255,0.1)',
                        tension: 0.4,
                        borderWidth: 2,
                    },
                    {
                        label: 'Actual Energy (kW)',
                        data: data.actual,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245,158,11,0.1)',
                        tension: 0.4,
                        borderWidth: 2,
                        borderDash: [5, 5],
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { labels: { color: '#f1f5f9' } } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' }, beginAtZero: false }
                }
            }
        });

        addEvent(`LSTM forecast complete — model: ${data.model}`, 'success');
    } catch (error) {
        addEvent('Forecast failed', 'error');
        console.error(error);
    } finally {
        document.getElementById('btn-forecast').disabled = false;
    }
}

// Historical Metrics
async function loadHistory() {
    try {
        const response = await fetch('/api/history?limit=200');
        const rows = await response.json();

        if (!rows.length) {
            addEvent('No history yet — start the simulation first', 'warning');
            return;
        }

        const labels = rows.map(r => `t${r.sim_time}`);
        const energy  = rows.map(r => r.total_energy);
        const cost    = rows.map(r => r.cost_per_hour);
        const carbon  = rows.map(r => r.carbon_emissions);
        const util    = rows.map(r => r.server_utilization);

        if (charts.history) charts.history.destroy();

        const ctx = document.getElementById('history-chart').getContext('2d');
        charts.history = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Energy (kWh)', data: energy, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.3, yAxisID: 'y' },
                    { label: 'Cost ($/h)',   data: cost,   borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.1)',  tension: 0.3, yAxisID: 'y' },
                    { label: 'Carbon (g)',   data: carbon, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)',  tension: 0.3, yAxisID: 'y1' },
                    { label: 'Utilization (%)', data: util, borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.1)', tension: 0.3, yAxisID: 'y1' },
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { labels: { color: '#f1f5f9' } } },
                scales: {
                    x:  { ticks: { color: '#94a3b8', maxTicksLimit: 20 }, grid: { color: '#334155' } },
                    y:  { type: 'linear', position: 'left',  ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                    y1: { type: 'linear', position: 'right', ticks: { color: '#94a3b8' }, grid: { display: false } },
                }
            }
        });

        addEvent(`History loaded — ${rows.length} data points`, 'success');
    } catch (error) {
        addEvent('Failed to load history', 'error');
        console.error(error);
    }
}

// Carbon Footprint Optimizer
async function runCarbonOptimizer() {
    document.getElementById('btn-carbon').disabled = true;
    try {
        const response = await fetch('/api/carbon');
        const data = await response.json();

        document.getElementById('carbon-naive').textContent = data.naive_co2.toFixed(1);
        document.getElementById('carbon-optimal').textContent = data.optimal_co2.toFixed(1);
        document.getElementById('carbon-saved').textContent = data.co2_saved.toFixed(1);
        document.getElementById('carbon-savings-pct').textContent = `${data.savings_pct.toFixed(1)}% savings`;

        // Build bar chart colored by hour type
        const colors = Array.from({ length: 24 }, (_, i) => {
            if (data.green_hours.includes(i)) return 'rgba(16,185,129,0.8)';
            if (data.peak_hours.includes(i)) return 'rgba(239,68,68,0.8)';
            return 'rgba(245,158,11,0.8)';
        });

        if (charts.carbon) {
            charts.carbon.destroy();
        }
        const ctx = document.getElementById('carbon-chart').getContext('2d');
        charts.carbon = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
                datasets: [{
                    label: 'Carbon Intensity (g CO₂/kWh)',
                    data: data.carbon_intensity,
                    backgroundColor: colors,
                    borderWidth: 0,
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { labels: { color: '#f1f5f9' } } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' }, beginAtZero: true }
                }
            }
        });

        document.getElementById('carbon-results').style.display = 'block';
        addEvent(`Carbon optimizer: ${data.savings_pct.toFixed(1)}% CO₂ reduction possible`, 'success');
    } catch (error) {
        addEvent('Carbon optimizer failed', 'error');
        console.error(error);
    } finally {
        document.getElementById('btn-carbon').disabled = false;
    }
}

// Quantum vs Classical Benchmark
async function runBenchmark() {
    document.getElementById('benchmark-results').style.display = 'none';
    document.getElementById('benchmark-loading').style.display = 'block';
    document.getElementById('btn-benchmark').disabled = true;

    try {
        const response = await fetch('/api/benchmark');
        const data = await response.json();

        const g = data.results.greedy;
        const q = data.results.qaoa;

        document.getElementById('bench-greedy-energy').textContent = g.energy.toFixed(4);
        document.getElementById('bench-greedy-time').textContent = g.time_ms + ' ms';
        document.getElementById('bench-greedy-util').textContent = (g.avg_utilization * 100).toFixed(1) + '%';
        document.getElementById('bench-greedy-balance').textContent = g.load_balance.toFixed(3);
        document.getElementById('bench-greedy-tasks').textContent = g.assigned_tasks;

        if (q && !q.error) {
            document.getElementById('bench-qaoa-energy').textContent = q.energy.toFixed(4);
            document.getElementById('bench-qaoa-time').textContent = q.time_ms + ' ms';
            document.getElementById('bench-qaoa-util').textContent = (q.avg_utilization * 100).toFixed(1) + '%';
            document.getElementById('bench-qaoa-balance').textContent = q.load_balance.toFixed(3);
            document.getElementById('bench-qaoa-tasks').textContent = q.assigned_tasks;
        } else {
            ['energy','time','util','balance','tasks'].forEach(k => {
                document.getElementById(`bench-qaoa-${k}`).textContent = q?.error ? 'Error' : '--';
            });
        }

        const imp = data.energy_improvement_pct;
        const badge = document.getElementById('bench-improvement');
        badge.textContent = (imp >= 0 ? '+' : '') + imp.toFixed(1) + '%';
        badge.style.background = imp > 0 ? 'var(--success-color)' : 'var(--danger-color)';

        addEvent(`Benchmark done — QAOA ${imp >= 0 ? 'saved' : 'used'} ${Math.abs(imp).toFixed(1)}% energy vs Greedy`, imp >= 0 ? 'success' : 'warning');
        document.getElementById('benchmark-results').style.display = 'grid';
    } catch (error) {
        addEvent('Benchmark failed', 'error');
        console.error(error);
    } finally {
        document.getElementById('benchmark-loading').style.display = 'none';
        document.getElementById('btn-benchmark').disabled = false;
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
