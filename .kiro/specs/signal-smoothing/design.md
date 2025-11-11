# Design Document

## Overview

The Calculus Signal Smoothing system is a Python-based application that demonstrates convolution-based signal smoothing through both a programmatic API and an interactive web interface. The system implements moving average and Gaussian filters with O(n) efficiency, provides real-time visualization capabilities, and includes educational content connecting discrete signal processing to calculus concepts.

The architecture consists of three main layers:
1. **Core Library Layer**: Pure Python functions for signal smoothing (src/smoothing.py)
2. **Web Application Layer**: Flask-based REST API and HTML/JavaScript frontend
3. **Demonstration Layer**: Jupyter notebook for educational exploration

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interfaces                       │
├──────────────────────┬──────────────────────────────────────┤
│   Web Browser        │      Jupyter Notebook                │
│   (HTML/JS/Plotly)   │      (demo.ipynb)                    │
└──────────┬───────────┴──────────────┬───────────────────────┘
           │                          │
           │ HTTP/JSON                │ Direct Import
           │                          │
┌──────────▼──────────────────────────▼───────────────────────┐
│              Application Layer                               │
├──────────────────────┬──────────────────────────────────────┤
│   Flask Web Server   │      Python API                      │
│   (app.py)           │                                      │
└──────────┬───────────┴──────────────────────────────────────┘
           │
           │ Function Calls
           │
┌──────────▼──────────────────────────────────────────────────┐
│              Core Library Layer                              │
├──────────────────────────────────────────────────────────────┤
│   src/smoothing.py                                           │
│   - moving_average()                                         │
│   - gaussian_kernel()                                        │
│   - gaussian_smooth()                                        │
└──────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- Python 3.8+
- NumPy 1.20+ (numerical computations)
- Flask 2.0+ (web framework)
- Flask-CORS (cross-origin support)

**Frontend:**
- HTML5/CSS3
- Vanilla JavaScript (ES6+)
- Plotly.js 2.0+ (interactive plotting)
- Modern CSS (Flexbox/Grid for layout)

**Development/Testing:**
- pytest (unit testing)
- Jupyter Notebook (demonstrations)
- Matplotlib (static plots in notebook)

## Components and Interfaces

### 1. Core Library (src/smoothing.py)

The core library provides three main functions with clean, functional interfaces.

#### 1.1 moving_average()

```python
def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute centered moving average using cumulative sum algorithm.
    
    Args:
        signal: 1D numpy array of float64 values
        window_size: Positive odd integer for window size
        
    Returns:
        Smoothed signal as 1D numpy array (same length as input)
        
    Raises:
        ValueError: If window_size is even or less than 1
        
    Algorithm:
        - Uses cumsum trick for O(n) complexity
        - Applies reflection padding for edge handling
        - Centered window: (window_size-1)//2 on each side
    """
```

**Implementation Strategy:**
- Convert input to float64 immediately
- Validate window_size (must be odd, >= 1)
- Apply reflection padding: `np.pad(signal, pad_width, mode='reflect')`
- Compute cumulative sum with prepended zero
- Extract windowed averages: `(cumsum[w:] - cumsum[:-w]) / w`
- Trim to original signal length

**Edge Case Handling:**
- Empty arrays: raise ValueError
- Window larger than signal: raise ValueError with helpful message
- Single-element signal: return copy of input

#### 1.2 gaussian_kernel()

```python
def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """
    Generate normalized 1D Gaussian kernel.
    
    Args:
        sigma: Standard deviation of Gaussian (must be > 0)
        truncate: Number of standard deviations to include (default 4.0)
        
    Returns:
        Normalized 1D numpy array where sum equals 1.0
        
    Algorithm:
        - Radius = ceil(truncate * sigma)
        - x values from -radius to +radius
        - Gaussian: exp(-x²/(2σ²))
        - Normalize: divide by sum
    """
```

**Implementation Strategy:**
- Validate sigma > 0
- Compute radius: `int(np.ceil(truncate * sigma))`
- Create x array: `np.arange(-radius, radius + 1)`
- Compute unnormalized kernel: `np.exp(-0.5 * (x / sigma) ** 2)`
- Normalize: `kernel / kernel.sum()`
- Ensure minimum kernel length of 3 for very small sigma

#### 1.3 gaussian_smooth()

```python
def gaussian_smooth(signal: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to signal.
    
    Args:
        signal: 1D numpy array
        sigma: Gaussian standard deviation
        truncate: Kernel truncation parameter
        
    Returns:
        Smoothed signal (same length as input)
        
    Algorithm:
        - Generate kernel using gaussian_kernel()
        - Apply reflection padding
        - Convolve using np.convolve()
        - Trim to original length
    """
```

**Implementation Strategy:**
- Generate kernel via `gaussian_kernel(sigma, truncate)`
- Compute padding: `pad_width = len(kernel) // 2`
- Pad signal with reflection mode
- Convolve: `np.convolve(padded_signal, kernel, mode='valid')`
- Return result (automatically correct length due to padding calculation)

### 2. Web Application (app.py + static files)

#### 2.1 Flask Application Structure

```python
# app.py
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
from src.smoothing import moving_average, gaussian_smooth

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """Serve main web interface"""
    
@app.route('/api/generate_signal', methods=['POST'])
def generate_signal():
    """Generate synthetic noisy signal"""
    
@app.route('/api/smooth', methods=['POST'])
def smooth_signal():
    """Apply smoothing filters to signal"""
```

#### 2.2 API Endpoints

**POST /api/generate_signal**

Request body:
```json
{
  "n_points": 2000,
  "frequency": 5.0,
  "noise_std": 0.6,
  "add_trend": false,
  "seed": 42
}
```

Response:
```json
{
  "time": [0.0, 0.001, 0.002, ...],
  "signal": [0.5, 0.52, 0.48, ...],
  "clean": [0.0, 0.031, 0.062, ...]
}
```

**POST /api/smooth**

Request body:
```json
{
  "signal": [0.5, 0.52, 0.48, ...],
  "filters": {
    "moving_average": {"window_size": 11},
    "gaussian": {"sigma": 2.5}
  }
}
```

Response:
```json
{
  "moving_average": {
    "result": [0.501, 0.503, ...],
    "compute_time_ms": 1.2
  },
  "gaussian": {
    "result": [0.499, 0.502, ...],
    "compute_time_ms": 2.5
  }
}
```

#### 2.3 Frontend Structure

**HTML Structure (templates/index.html):**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Signal Smoothing Visualization</title>
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Signal Smoothing by Convolution</h1>
        </header>
        
        <div class="main-layout">
            <aside class="controls-panel">
                <!-- Signal generation controls -->
                <!-- Filter controls -->
                <!-- Action buttons -->
            </aside>
            
            <main class="visualization-panel">
                <div id="plot"></div>
                <div id="metrics"></div>
            </main>
        </div>
    </div>
    
    <script src="/static/js/app.js"></script>
</body>
</html>
```

**JavaScript Architecture (static/js/app.js):**

```javascript
// State management
const state = {
    signalParams: {
        n_points: 2000,
        frequency: 5.0,
        noise_std: 0.6,
        add_trend: false,
        seed: 42
    },
    filterParams: {
        moving_average: { enabled: true, window_size: 11 },
        gaussian: { enabled: true, sigma: 2.5 }
    },
    currentData: null
};

// API communication
async function generateSignal() { ... }
async function applyFilters() { ... }

// UI updates
function updatePlot(data) { ... }
function updateMetrics(metrics) { ... }

// Event handlers
function setupEventListeners() { ... }
```

### 3. Jupyter Notebook (notebooks/demo.ipynb)

**Notebook Structure:**

1. **Introduction Cell** (Markdown)
   - Overview of signal smoothing
   - Calculus connections

2. **Setup Cell** (Code)
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from src.smoothing import moving_average, gaussian_smooth, gaussian_kernel
   ```

3. **Signal Generation Cell** (Code)
   - Create synthetic noisy sine wave
   - Visualize original signal

4. **Moving Average Demo Cell** (Code)
   - Apply moving average with different window sizes
   - Plot comparison
   - Discuss O(n) complexity

5. **Gaussian Smoothing Demo Cell** (Code)
   - Show Gaussian kernel shape
   - Apply Gaussian smoothing with different sigmas
   - Plot comparison

6. **Parameter Sweep Cell** (Code)
   - Compute smoothing quality vs parameters
   - Create parameter sweep plots

7. **Calculus Connection Cell** (Markdown + Code)
   - Explain convolution as integral
   - Show derivative smoothing
   - Demonstrate Riemann sum approximation

## Data Models

### Signal Data Structure

Signals are represented as NumPy arrays with associated metadata:

```python
@dataclass
class SignalData:
    """Container for signal data and metadata"""
    time: np.ndarray          # Time points (1D array)
    values: np.ndarray        # Signal values (1D array)
    sample_rate: float        # Samples per second
    metadata: dict            # Additional info (frequency, noise level, etc.)
```

For web API, signals are serialized as JSON arrays with metadata in separate fields.

### Filter Configuration

```python
@dataclass
class FilterConfig:
    """Configuration for smoothing filters"""
    filter_type: str          # "moving_average" or "gaussian"
    parameters: dict          # Type-specific parameters
    enabled: bool = True
```

## Error Handling

### Core Library Error Handling

**Input Validation:**
- Check array dimensions (must be 1D)
- Check parameter ranges (window_size > 0, sigma > 0)
- Check array lengths (signal must be longer than window)

**Error Types:**
```python
# Invalid input shape
raise ValueError("Signal must be 1D array")

# Invalid parameter
raise ValueError("Window size must be positive odd integer")

# Numerical issues
if not np.all(np.isfinite(signal)):
    raise ValueError("Signal contains NaN or infinite values")
```

### Web Application Error Handling

**API Error Responses:**
```json
{
  "error": "Invalid parameter",
  "message": "Window size must be between 3 and 101",
  "code": "INVALID_PARAMETER"
}
```

**Frontend Error Handling:**
- Display user-friendly error messages in UI
- Provide fallback to previous valid state
- Log errors to browser console for debugging
- Show loading indicators during async operations

**HTTP Status Codes:**
- 200: Success
- 400: Invalid request parameters
- 500: Server error (computation failure)

## Testing Strategy

### Unit Tests (tests/test_smoothing.py)

**Test Categories:**

1. **Correctness Tests**
   ```python
   def test_moving_average_constant_signal():
       """Moving average of constant signal should return constant"""
       
   def test_gaussian_kernel_normalization():
       """Gaussian kernel should sum to 1.0"""
       
   def test_output_length():
       """Output should have same length as input"""
   ```

2. **Edge Case Tests**
   ```python
   def test_small_signals():
       """Handle signals shorter than typical windows"""
       
   def test_extreme_parameters():
       """Handle very large/small parameter values"""
   ```

3. **Numerical Accuracy Tests**
   ```python
   def test_numerical_precision():
       """Verify float64 precision maintained"""
       
   def test_known_results():
       """Compare against analytically computed results"""
   ```

### Integration Tests

**Web API Tests:**
```python
def test_generate_signal_endpoint():
    """Test signal generation API"""
    
def test_smooth_endpoint():
    """Test smoothing API with various parameters"""
    
def test_error_responses():
    """Verify proper error handling"""
```

### Manual Testing Checklist

- [ ] Web interface loads without errors
- [ ] All sliders update plot in real-time
- [ ] Parameter ranges are enforced
- [ ] Download button produces valid PNG
- [ ] Reset button restores defaults
- [ ] Metrics display correct values
- [ ] Works in Chrome, Firefox, Safari, Edge
- [ ] Responsive layout on different screen sizes
- [ ] Notebook executes all cells successfully

## Performance Considerations

### Computational Complexity

**Moving Average:**
- Time: O(n) using cumsum trick
- Space: O(n) for padded array
- Typical performance: <1ms for n=2000

**Gaussian Smoothing:**
- Time: O(n × m) where m = kernel length
- For σ=2.5, truncate=4.0: m ≈ 21
- Space: O(n + m)
- Typical performance: <5ms for n=2000

### Web Application Performance

**Optimization Strategies:**
1. **Debouncing**: Delay filter recomputation during slider drag
   ```javascript
   let debounceTimer;
   slider.addEventListener('input', () => {
       clearTimeout(debounceTimer);
       debounceTimer = setTimeout(applyFilters, 200);
   });
   ```

2. **Caching**: Store generated signal to avoid regeneration
3. **Lazy Loading**: Only compute enabled filters
4. **Response Compression**: Use gzip for JSON responses

**Target Performance:**
- Signal generation: <50ms
- Filter application: <100ms
- Plot update: <200ms
- Total interaction latency: <500ms

### Memory Management

- Limit signal length to 10,000 points in web interface
- Clear old plot data before creating new plots
- Use NumPy views instead of copies where possible

## Deployment Considerations

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run web server
python app.py

# Access at http://localhost:5000
```

### Production Deployment (Optional)

For production deployment, consider:
- Use Gunicorn or uWSGI as WSGI server
- Add nginx reverse proxy
- Enable HTTPS
- Set up proper logging
- Configure CORS appropriately
- Add rate limiting

**Example Gunicorn command:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## UI/UX Design Specifications

### Color Scheme

```css
:root {
    --primary-color: #2563eb;      /* Blue for primary actions */
    --secondary-color: #7c3aed;    /* Purple for secondary */
    --success-color: #10b981;      /* Green for success states */
    --background: #f8fafc;         /* Light gray background */
    --surface: #ffffff;            /* White for cards/panels */
    --text-primary: #1e293b;       /* Dark gray for text */
    --text-secondary: #64748b;     /* Medium gray for labels */
    --border: #e2e8f0;             /* Light border color */
}
```

### Layout Structure

**Desktop (≥1024px):**
- Two-column layout: 300px sidebar + flexible main area
- Controls panel on left with fixed width
- Visualization panel on right taking remaining space

**Tablet (768px-1023px):**
- Single column with controls above plot
- Collapsible controls section

### Typography

- Headings: System font stack (SF Pro, Segoe UI, Roboto)
- Body: 16px base size for readability
- Code/numbers: Monospace font for parameter values

### Interactive Elements

**Sliders:**
- Clear labels with current value display
- Smooth dragging with visual feedback
- Snap to valid values (e.g., odd integers for window size)

**Buttons:**
- Primary button: Filled with primary color
- Secondary button: Outlined
- Hover states with subtle transitions

**Plot:**
- Responsive sizing
- Plotly modebar for zoom/pan/download
- Legend with toggle capability
- Axis labels with units

## Calculus Connection Documentation

The README.md will include a dedicated section explaining the mathematical foundations:

### Convolution as Integral

Continuous convolution:
```
(f * g)(t) = ∫ f(τ)g(t-τ) dτ
```

Discrete approximation:
```
(f * g)[n] = Σ f[k]g[n-k]
```

### Smoothing and Derivatives

Smoothing reduces high-frequency components, which correspond to rapid changes (high derivatives). The smoothed signal has smaller second derivatives, making it "smoother" in the calculus sense.

### Riemann Sums

The moving average is a Riemann sum approximation:
```
Average ≈ (1/Δt) ∫ f(t) dt ≈ (1/n) Σ f[i]
```

## Future Enhancements

Potential features for future versions:

1. **Additional Filters**: Median filter, Savitzky-Golay filter
2. **Signal Upload**: Allow users to upload their own CSV data
3. **Comparison Mode**: Side-by-side comparison of multiple parameter sets
4. **Animation Export**: Generate animated GIFs of parameter sweeps
5. **Frequency Domain View**: Add FFT magnitude plots
6. **Mobile Support**: Responsive design for phones
7. **Preset Signals**: Library of example signals (ECG, audio, etc.)
8. **Performance Metrics**: Show detailed timing breakdowns
9. **Export Data**: Download filtered signals as CSV
10. **Collaborative Features**: Share parameter configurations via URL

## Dependencies and Versions

```
# requirements.txt
numpy>=1.20.0,<2.0.0
matplotlib>=3.3.0,<4.0.0
flask>=2.0.0,<3.0.0
flask-cors>=3.0.0,<4.0.0
pytest>=7.0.0,<8.0.0
jupyter>=1.0.0,<2.0.0
```

## File Structure Summary

```
calculus-signal-smoothing/
├── README.md                    # Project overview and calculus explanation
├── requirements.txt             # Python dependencies
├── app.py                       # Flask web application
│
├── src/
│   ├── __init__.py
│   └── smoothing.py             # Core filter implementations
│
├── static/
│   ├── css/
│   │   └── style.css            # Web interface styling
│   └── js/
│       └── app.js               # Frontend JavaScript
│
├── templates/
│   └── index.html               # Main web interface
│
├── notebooks/
│   └── demo.ipynb               # Jupyter demonstration
│
├── tests/
│   └── test_smoothing.py        # Unit tests
│
├── plots/                       # Generated plots directory
│   └── .gitkeep
│
└── data/                        # Optional sample data
    └── sample_signal.npy
```
