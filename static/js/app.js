/**
 * Signal Smoothing Web Application
 * 
 * Interactive visualization tool for demonstrating moving average and
 * Gaussian smoothing filters on synthetic noisy signals.
 */

// ============================================================================
// State Management
// ============================================================================

const DEFAULT_STATE = {
    signalParams: {
        n_points: 2000,
        frequency: 5.0,
        noise_std: 0.6,
        add_trend: false,
        seed: 42
    },
    filterParams: {
        moving_average: {
            enabled: true,
            window_size: 11
        },
        gaussian: {
            enabled: true,
            sigma: 2.5
        }
    },
    currentData: null
};

// Deep clone the default state
const state = JSON.parse(JSON.stringify(DEFAULT_STATE));

// Debounce timer for parameter changes
let debounceTimer = null;


// ============================================================================
// API Communication Functions
// ============================================================================

/**
 * Generate a new signal from the backend API
 */
async function generateSignal() {
    try {
        const response = await fetch('/api/generate_signal', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(state.signalParams)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Failed to generate signal');
        }
        
        const data = await response.json();
        state.currentData = data;
        return data;
    } catch (error) {
        console.error('Error generating signal:', error);
        alert(`Error generating signal: ${error.message}`);
        throw error;
    }
}

/**
 * Apply smoothing filters to the current signal
 */
async function applyFilters() {
    if (!state.currentData) {
        return null;
    }
    
    try {
        // Build filters object with only enabled filters
        const filters = {};
        
        if (state.filterParams.moving_average.enabled) {
            filters.moving_average = {
                window_size: state.filterParams.moving_average.window_size
            };
        }
        
        if (state.filterParams.gaussian.enabled) {
            filters.gaussian = {
                sigma: state.filterParams.gaussian.sigma
            };
        }
        
        // If no filters enabled, return empty results
        if (Object.keys(filters).length === 0) {
            return {};
        }
        
        const response = await fetch('/api/smooth', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                signal: state.currentData.signal,
                filters: filters
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Failed to apply filters');
        }
        
        const results = await response.json();
        return results;
    } catch (error) {
        console.error('Error applying filters:', error);
        alert(`Error applying filters: ${error.message}`);
        throw error;
    }
}


// ============================================================================
// Visualization Functions
// ============================================================================

/**
 * Update the Plotly visualization with current data
 */
function updatePlot(data, smoothedData) {
    const traces = [];
    
    // Noisy signal trace
    traces.push({
        x: data.time,
        y: data.signal,
        mode: 'markers',
        type: 'scatter',
        name: 'Noisy Signal',
        marker: {
            size: 3,
            color: '#94a3b8',
            opacity: 0.6
        }
    });
    
    // Clean signal trace (reference)
    traces.push({
        x: data.time,
        y: data.clean,
        mode: 'lines',
        type: 'scatter',
        name: 'Clean Signal',
        line: {
            color: '#cbd5e1',
            width: 1,
            dash: 'dash'
        }
    });
    
    // Moving average trace
    if (smoothedData.moving_average && state.filterParams.moving_average.enabled) {
        traces.push({
            x: data.time,
            y: smoothedData.moving_average.result,
            mode: 'lines',
            type: 'scatter',
            name: `Moving Average (w=${state.filterParams.moving_average.window_size})`,
            line: {
                color: '#2563eb',
                width: 2
            }
        });
    }
    
    // Gaussian trace
    if (smoothedData.gaussian && state.filterParams.gaussian.enabled) {
        traces.push({
            x: data.time,
            y: smoothedData.gaussian.result,
            mode: 'lines',
            type: 'scatter',
            name: `Gaussian (σ=${state.filterParams.gaussian.sigma})`,
            line: {
                color: '#7c3aed',
                width: 2
            }
        });
    }
    
    // Layout configuration
    const layout = {
        title: {
            text: 'Signal Smoothing Comparison',
            font: { size: 20 }
        },
        xaxis: {
            title: 'Time (seconds)',
            gridcolor: '#e2e8f0'
        },
        yaxis: {
            title: 'Amplitude',
            gridcolor: '#e2e8f0'
        },
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255, 255, 255, 0.9)',
            bordercolor: '#e2e8f0',
            borderwidth: 1
        },
        plot_bgcolor: '#ffffff',
        paper_bgcolor: '#ffffff',
        hovermode: 'closest',
        autosize: true,
        margin: { l: 60, r: 40, t: 60, b: 60 }
    };
    
    // Configuration
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    };
    
    Plotly.newPlot('plot', traces, layout, config);
}

/**
 * Update the metrics display panel
 */
function updateMetrics(smoothedData) {
    const metricsDiv = document.getElementById('metrics');
    
    if (!smoothedData || Object.keys(smoothedData).length === 0) {
        metricsDiv.innerHTML = '<p style="color: #64748b; text-align: center;">No filters enabled</p>';
        return;
    }
    
    let html = '<h3>Performance Metrics</h3>';
    html += '<table class="metrics-table">';
    html += '<thead><tr><th>Filter</th><th>Compute Time</th></tr></thead>';
    html += '<tbody>';
    
    if (smoothedData.moving_average) {
        html += '<tr>';
        html += '<td>Moving Average</td>';
        html += `<td>${smoothedData.moving_average.compute_time_ms.toFixed(3)} ms</td>`;
        html += '</tr>';
    }
    
    if (smoothedData.gaussian) {
        html += '<tr>';
        html += '<td>Gaussian</td>';
        html += `<td>${smoothedData.gaussian.compute_time_ms.toFixed(3)} ms</td>`;
        html += '</tr>';
    }
    
    html += '</tbody></table>';
    metricsDiv.innerHTML = html;
}


// ============================================================================
// Event Handlers
// ============================================================================

/**
 * Set up all event listeners for UI controls
 */
function setupEventListeners() {
    // Signal generation controls
    const frequencySlider = document.getElementById('frequency');
    const frequencyValue = document.getElementById('frequency-value');
    const noiseStdSlider = document.getElementById('noise-std');
    const noiseStdValue = document.getElementById('noise-std-value');
    const addTrendCheckbox = document.getElementById('add-trend');
    
    // Filter controls
    const maEnabledCheckbox = document.getElementById('ma-enabled');
    const windowSizeSlider = document.getElementById('window-size');
    const windowSizeValue = document.getElementById('window-size-value');
    const gaussEnabledCheckbox = document.getElementById('gauss-enabled');
    const sigmaSlider = document.getElementById('sigma');
    const sigmaValue = document.getElementById('sigma-value');
    
    // Action buttons
    const generateButton = document.getElementById('btn-generate');
    const downloadButton = document.getElementById('btn-download');
    const resetButton = document.getElementById('btn-reset');
    
    // Frequency slider
    frequencySlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        frequencyValue.textContent = value.toFixed(1);
        state.signalParams.frequency = value;
    });
    
    frequencySlider.addEventListener('change', () => {
        debounceUpdate(async () => {
            await generateSignal();
            const smoothed = await applyFilters();
            updatePlot(state.currentData, smoothed);
            updateMetrics(smoothed);
        });
    });
    
    // Noise std slider
    noiseStdSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        noiseStdValue.textContent = value.toFixed(1);
        state.signalParams.noise_std = value;
    });
    
    noiseStdSlider.addEventListener('change', () => {
        debounceUpdate(async () => {
            await generateSignal();
            const smoothed = await applyFilters();
            updatePlot(state.currentData, smoothed);
            updateMetrics(smoothed);
        });
    });
    
    // Add trend checkbox
    addTrendCheckbox.addEventListener('change', async (e) => {
        state.signalParams.add_trend = e.target.checked;
        await generateSignal();
        const smoothed = await applyFilters();
        updatePlot(state.currentData, smoothed);
        updateMetrics(smoothed);
    });
    
    // Moving average enabled checkbox
    maEnabledCheckbox.addEventListener('change', async (e) => {
        state.filterParams.moving_average.enabled = e.target.checked;
        const smoothed = await applyFilters();
        updatePlot(state.currentData, smoothed);
        updateMetrics(smoothed);
    });
    
    // Window size slider
    windowSizeSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        windowSizeValue.textContent = value;
        state.filterParams.moving_average.window_size = value;
    });
    
    windowSizeSlider.addEventListener('change', () => {
        debounceUpdate(async () => {
            const smoothed = await applyFilters();
            updatePlot(state.currentData, smoothed);
            updateMetrics(smoothed);
        });
    });
    
    // Gaussian enabled checkbox
    gaussEnabledCheckbox.addEventListener('change', async (e) => {
        state.filterParams.gaussian.enabled = e.target.checked;
        const smoothed = await applyFilters();
        updatePlot(state.currentData, smoothed);
        updateMetrics(smoothed);
    });
    
    // Sigma slider
    sigmaSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        sigmaValue.textContent = value.toFixed(1);
        state.filterParams.gaussian.sigma = value;
    });
    
    sigmaSlider.addEventListener('change', () => {
        debounceUpdate(async () => {
            const smoothed = await applyFilters();
            updatePlot(state.currentData, smoothed);
            updateMetrics(smoothed);
        });
    });
    
    // Generate button
    generateButton.addEventListener('click', async () => {
        await generateSignal();
        const smoothed = await applyFilters();
        updatePlot(state.currentData, smoothed);
        updateMetrics(smoothed);
    });
    
    // Download button
    downloadButton.addEventListener('click', () => {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const filename = `signal-smoothing-${timestamp}.png`;
        
        Plotly.downloadImage('plot', {
            format: 'png',
            width: 1200,
            height: 800,
            filename: filename
        });
    });
    
    // Reset button
    resetButton.addEventListener('click', async () => {
        // Reset state to defaults
        Object.assign(state, JSON.parse(JSON.stringify(DEFAULT_STATE)));
        
        // Update UI controls
        frequencySlider.value = DEFAULT_STATE.signalParams.frequency;
        frequencyValue.textContent = DEFAULT_STATE.signalParams.frequency.toFixed(1);
        
        noiseStdSlider.value = DEFAULT_STATE.signalParams.noise_std;
        noiseStdValue.textContent = DEFAULT_STATE.signalParams.noise_std.toFixed(1);
        
        addTrendCheckbox.checked = DEFAULT_STATE.signalParams.add_trend;
        
        maEnabledCheckbox.checked = DEFAULT_STATE.filterParams.moving_average.enabled;
        windowSizeSlider.value = DEFAULT_STATE.filterParams.moving_average.window_size;
        windowSizeValue.textContent = DEFAULT_STATE.filterParams.moving_average.window_size;
        
        gaussEnabledCheckbox.checked = DEFAULT_STATE.filterParams.gaussian.enabled;
        sigmaSlider.value = DEFAULT_STATE.filterParams.gaussian.sigma;
        sigmaValue.textContent = DEFAULT_STATE.filterParams.gaussian.sigma.toFixed(1);
        
        // Regenerate signal and plot
        await generateSignal();
        const smoothed = await applyFilters();
        updatePlot(state.currentData, smoothed);
        updateMetrics(smoothed);
    });
}

/**
 * Debounce function to delay updates during slider drag
 */
function debounceUpdate(callback) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(callback, 200);
}


// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize the application
 */
async function init() {
    console.log('Initializing Signal Smoothing Application...');
    
    // Set up event listeners
    setupEventListeners();
    
    // Generate initial signal
    await generateSignal();
    
    // Apply initial filters
    const smoothed = await applyFilters();
    
    // Render initial plot
    updatePlot(state.currentData, smoothed);
    
    // Update metrics
    updateMetrics(smoothed);
    
    console.log('Application initialized successfully');
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
