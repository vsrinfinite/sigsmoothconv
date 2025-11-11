# Signal Smoothing by Convolution

An interactive Python application demonstrating convolution-based signal smoothing using moving average and Gaussian filters. This project provides both a web-based visualization tool and a Jupyter notebook for educational exploration of signal processing concepts and their connections to calculus.

## Overview

Signal smoothing is a fundamental operation in signal processing that removes noise while preserving important signal features. This project implements two classic smoothing filters:

- **Moving Average Filter**: Fast O(n) smoothing using cumulative sum algorithm
- **Gaussian Filter**: Smooth, continuous filtering with controlled frequency response

Both filters work by convolving the input signal with a kernel, demonstrating how discrete operations approximate continuous calculus concepts.

## Calculus Connection

### Convolution as Integral

Continuous convolution is defined as an integral:

```
(f * g)(t) = ∫ f(τ)g(t-τ) dτ
```

The discrete convolution used in signal processing approximates this integral:

```
(f * g)[n] = Σ f[k]g[n-k]
```

This discrete sum is a Riemann sum approximation of the continuous integral, where the sampling interval Δt acts as the width of each rectangle in the Riemann sum.

### Smoothing and Derivatives

Smoothing filters attenuate high-frequency components in a signal. In calculus terms, high frequencies correspond to rapid changes—large derivatives. When we smooth a signal:

1. **First derivatives** (rate of change) become smaller in magnitude
2. **Second derivatives** (curvature) are dramatically reduced
3. The signal becomes "smoother" in the mathematical sense

The smoothed signal has smaller higher-order derivatives, making it more continuous and less jagged.

### Riemann Sums

The moving average can be viewed as a Riemann sum approximation of an integral:

```
Average ≈ (1/Δt) ∫ f(t) dt ≈ (1/n) Σ f[i]
```

Each sample point represents a rectangle in the Riemann sum, and the average approximates the integral divided by the interval width.

## Features

- **Interactive Web Application**: Real-time parameter adjustment with instant visualization
- **Jupyter Notebook**: Educational demonstrations with detailed explanations
- **Efficient Algorithms**: O(n) moving average using cumulative sum trick
- **Publication-Quality Plots**: High-resolution visualizations suitable for presentations
- **Comprehensive Testing**: Unit and integration tests ensuring correctness
- **Modern UI**: Clean, responsive design with intuitive controls

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd signal-smoothing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- Flask >= 2.0.0
- Flask-CORS >= 3.0.0
- pytest >= 7.0.0
- Jupyter >= 1.0.0

## Usage

### Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser to:
```
http://localhost:5000
```

The web interface provides:
- Real-time signal generation with adjustable frequency and noise
- Interactive sliders for filter parameters
- Live visualization updates
- Performance metrics display
- Plot download functionality

### Jupyter Notebook

Launch Jupyter and open the demonstration notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook includes:
- Signal generation and visualization
- Moving average demonstrations with different window sizes
- Gaussian filter examples with various sigma values
- Calculus connections and derivative analysis
- Comparison plots and performance metrics

### Python API

Use the smoothing functions directly in your code:

```python
import numpy as np
from src.smoothing import moving_average, gaussian_smooth, gaussian_kernel

# Generate a noisy signal
t = np.linspace(0, 2, 1000)
signal = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, 1000)

# Apply moving average
smoothed_ma = moving_average(signal, window_size=11)

# Apply Gaussian smoothing
smoothed_gauss = gaussian_smooth(signal, sigma=2.5)

# Generate a custom Gaussian kernel
kernel = gaussian_kernel(sigma=3.0, truncate=4.0)
```

## Project Structure

```
signal-smoothing/
├── README.md                    # This file
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
│   ├── test_smoothing.py        # Unit tests for filters
│   └── test_api.py              # Integration tests for API
│
├── plots/                       # Generated plots directory
│   └── .gitkeep
│
└── data/                        # Optional sample data
    └── .gitkeep
```

## Testing

Run all tests with pytest:

```bash
pytest
```

Run specific test files:

```bash
pytest tests/test_smoothing.py -v
pytest tests/test_api.py -v
```

The test suite includes:
- 30 unit tests for core smoothing functions
- 16 integration tests for API endpoints
- Edge case and error handling tests
- Numerical precision verification

## Dependencies

Core dependencies:
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Static plotting in Jupyter notebook
- **Flask**: Web framework for backend API
- **Flask-CORS**: Cross-origin resource sharing support
- **Plotly.js**: Interactive plotting in web interface (CDN)
- **pytest**: Testing framework
- **Jupyter**: Interactive notebook environment

## Algorithm Details

### Moving Average

Time Complexity: O(n)  
Space Complexity: O(n)

The moving average uses the cumulative sum trick:
1. Compute cumulative sum of padded signal
2. Extract windowed sums: `cumsum[w:] - cumsum[:-w]`
3. Divide by window size
4. Trim to original length

### Gaussian Filter

Time Complexity: O(n × m) where m is kernel length  
Space Complexity: O(n + m)

The Gaussian filter:
1. Generates normalized Gaussian kernel
2. Applies reflection padding to signal
3. Performs convolution using NumPy
4. Returns result with original length

## Performance

Typical performance on a modern desktop (n=2000 points):
- Moving average (window=11): < 1 ms
- Gaussian smoothing (σ=2.5): < 5 ms
- Signal generation: < 50 ms
- Total interaction latency: < 500 ms

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional filter types (median, Savitzky-Golay)
- Signal upload functionality
- FFT-based convolution for large kernels
- Mobile-responsive design improvements
- Additional visualization options

## License

[Specify your license here]

## Acknowledgments

This project demonstrates fundamental signal processing concepts and their connections to calculus, making it suitable for educational purposes in signal processin
g, numerical methods, and applied mathematics courses.

## Example Output

When you run the web application or notebook, you'll see:

1. **Original Signal**: Noisy sinusoidal signal with Gaussian noise
2. **Clean Reference**: The underlying clean signal for comparison
3. **Moving Average Result**: Smoothed signal using window-based averaging
4. **Gaussian Result**: Smoothed signal using Gaussian kernel
5. **Performance Metrics**: Computation times for each filter

The visualizations clearly show how different filter parameters affect the trade-off between noise reduction and signal preservation.

## Technical Notes

- All computations use float64 precision for numerical accuracy
- Reflection padding minimizes edge artifacts
- Odd window sizes ensure centered averaging
- Gaussian kernels are normalized to sum to 1.0
- Web API validates all parameters and returns descriptive errors
- Frontend uses debouncing for smooth slider interactions

## Contact

For questions, issues, or suggestions, please open an issue on the project repository.
