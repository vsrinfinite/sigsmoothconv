# Implementation Plan

- [x] 1. Set up project structure and dependencies


  - Create directory structure (src/, static/, templates/, notebooks/, tests/, plots/, data/)
  - Create requirements.txt with numpy, matplotlib, flask, flask-cors, pytest, jupyter
  - Create empty __init__.py files in src/ and tests/
  - Create .gitignore for Python projects (exclude __pycache__, .pytest_cache, *.pyc, plots/*.png)
  - _Requirements: 20, 22_

- [ ] 2. Implement core smoothing functions in src/smoothing.py
  - _Requirements: 1, 2, 3, 14, 21_

- [x] 2.1 Implement moving_average() function


  - Write function signature with type hints: `moving_average(signal: np.ndarray, window_size: int) -> np.ndarray`
  - Add input validation (check 1D array, window_size is positive odd integer, signal length > window_size)
  - Convert input signal to float64
  - Implement reflection padding using np.pad()
  - Implement cumsum algorithm: prepend zero, compute cumsum, extract windowed averages
  - Trim result to original signal length
  - Add comprehensive docstring explaining algorithm, parameters, returns, and raises
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2.2 Implement gaussian_kernel() function


  - Write function signature: `gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray`
  - Add input validation (sigma > 0)
  - Compute radius as ceiling(truncate * sigma)
  - Create x array from -radius to +radius
  - Compute Gaussian values using exp(-x²/(2σ²))
  - Normalize kernel to sum to 1.0
  - Handle edge case of very small sigma (ensure minimum kernel length of 3)
  - Add docstring with formula and parameters
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_



- [ ] 2.3 Implement gaussian_smooth() function
  - Write function signature: `gaussian_smooth(signal: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray`
  - Convert input signal to float64
  - Generate kernel using gaussian_kernel()
  - Compute padding width as len(kernel) // 2
  - Apply reflection padding to signal
  - Perform convolution using np.convolve() with mode='valid'


  - Add docstring explaining the smoothing process
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 2.4 Write unit tests for core smoothing functions
  - Create tests/test_smoothing.py with pytest imports
  - Test moving_average with constant signal (should return constant within 1e-12)
  - Test moving_average output length equals input length
  - Test moving_average with known simple case (e.g., [1,2,3,4,5] with window 3)
  - Test gaussian_kernel normalization (sum should equal 1.0 within 1e-12)
  - Test gaussian_kernel symmetry (kernel should be symmetric)
  - Test gaussian_smooth output length equals input length
  - Test invalid inputs raise appropriate ValueErrors (negative window, even window, sigma <= 0)
  - Test edge case: signal shorter than window raises ValueError
  - Test numerical precision: results use float64


  - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

- [ ] 3. Implement Flask web application backend
  - _Requirements: 6, 7, 8, 9, 10, 14, 15, 21_

- [ ] 3.1 Create app.py with Flask application setup
  - Import Flask, jsonify, request, render_template
  - Import CORS from flask_cors


  - Import numpy and smoothing functions from src.smoothing
  - Create Flask app instance
  - Enable CORS on app
  - Add main route @app.route('/') that renders index.html
  - Add if __name__ == '__main__' block to run app on port 5000 with debug=True
  - _Requirements: 14.1, 14.2_

- [ ] 3.2 Implement /api/generate_signal endpoint
  - Create POST endpoint @app.route('/api/generate_signal', methods=['POST'])
  - Parse JSON request body for parameters: n_points, frequency, noise_std, add_trend, seed


  - Set numpy random seed if provided
  - Generate time array: np.linspace(0, duration, n_points)
  - Generate clean sinusoidal signal: np.sin(2 * pi * frequency * time)
  - Add Gaussian noise: clean + np.random.normal(0, noise_std, n_points)
  - Optionally add linear trend if add_trend is True
  - Return JSON response with time, signal (noisy), and clean arrays as lists
  - Add error handling for invalid parameters (return 400 status with error message)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 14.3_

- [ ] 3.3 Implement /api/smooth endpoint
  - Create POST endpoint @app.route('/api/smooth', methods=['POST'])
  - Parse JSON request body for signal array and filters dictionary
  - Convert signal list to numpy array


  - Initialize results dictionary
  - For each enabled filter in request (moving_average, gaussian):
    - Record start time using time.perf_counter()
    - Apply appropriate smoothing function with provided parameters
    - Record end time and compute duration in milliseconds
    - Store result array (as list) and compute_time_ms in results dictionary
  - Return JSON response with results for each filter
  - Add error handling for computation errors (return 500 status with error message)
  - Validate parameter ranges (window_size 3-101, sigma 0.5-10.0)
  - _Requirements: 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 10.1, 10.2, 10.3, 14.3, 14.4, 21.4_

- [x] 3.4 Add integration tests for API endpoints


  - Create tests/test_api.py with Flask test client
  - Test /api/generate_signal returns valid JSON with correct keys
  - Test /api/generate_signal with various parameter combinations
  - Test /api/smooth returns smoothed signals with correct length
  - Test /api/smooth with only moving_average enabled
  - Test /api/smooth with only gaussian enabled
  - Test error responses for invalid parameters (400 status)
  - Test compute_time_ms is present and positive
  - _Requirements: 14.3, 14.4_

- [x] 4. Implement web frontend interface

  - _Requirements: 6, 7, 8, 9, 10, 11, 12, 13, 15_

- [ ] 4.1 Create HTML structure in templates/index.html
  - Create HTML5 boilerplate with DOCTYPE, html, head, body
  - Add title: "Signal Smoothing Visualization"
  - Link to Plotly.js CDN (version 2.26.0 or later)
  - Link to static/css/style.css
  - Create header with h1 title and brief description

  - Create main container div with class "container"
  - Create two-column layout: aside.controls-panel and main.visualization-panel
  - In controls panel, add sections for: Signal Generation, Moving Average Filter, Gaussian Filter, Actions
  - In visualization panel, add div#plot for Plotly chart and div#metrics for performance metrics
  - Add script tag linking to static/js/app.js at end of body
  - _Requirements: 6.1, 11.1, 11.2, 11.4_

- [x] 4.2 Add signal generation controls to HTML

  - Create fieldset for "Signal Generation" in controls panel
  - Add range input for frequency (1-20, step 0.5, default 5.0) with label and value display span
  - Add range input for noise_std (0.0-2.0, step 0.1, default 0.6) with label and value display
  - Add checkbox for add_trend (default unchecked) with label
  - Add button "Generate New Signal" with id="btn-generate"
  - Add hidden input for seed (fixed value 42)


  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 4.3 Add filter controls to HTML
  - Create fieldset for "Moving Average Filter"
  - Add checkbox to enable/disable moving average (default checked) with label
  - Add range input for window_size (3-101, step 2, default 11) with label and value display
  - Create fieldset for "Gaussian Filter"
  - Add checkbox to enable/disable Gaussian (default checked) with label
  - Add range input for sigma (0.5-10.0, step 0.1, default 2.5) with label and value display
  - _Requirements: 8.1, 8.2, 8.4, 8.5, 9.1, 9.2, 9.4, 9.5_

- [ ] 4.4 Add action buttons to HTML
  - Create div for action buttons
  - Add button "Download Plot" with id="btn-download"
  - Add button "Reset to Defaults" with id="btn-reset"
  - Style buttons with appropriate classes
  - _Requirements: 12.1, 13.1_

- [x] 4.5 Create CSS styling in static/css/style.css


  - Define CSS custom properties for color scheme (primary, secondary, background, surface, text colors, borders)
  - Reset default margins and set box-sizing: border-box
  - Style body with background color and font family (system font stack)
  - Style .container with max-width and centered layout
  - Style header with padding and border-bottom

  - Create two-column layout using flexbox or grid (.main-layout)
  - Style .controls-panel with fixed width (300px), padding, background, border
  - Style .visualization-panel to take remaining space
  - Style fieldsets with margin, padding, border-radius
  - Style labels to display block with margin
  - Style range inputs with full width and custom track/thumb styling
  - Style buttons with padding, border-radius, colors, hover states, transitions
  - Style #plot div with minimum height (600px) and border

  - Style #metrics div with padding and background
  - Add responsive styles for tablet sizes (collapse to single column)
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 4.6 Implement JavaScript application logic in static/js/app.js
  - _Requirements: 6.2, 6.3, 6.4, 7.4, 8.3, 9.3, 10.4, 12.2, 12.3, 12.4, 12.5, 13.2, 13.3, 13.4, 13.5, 15.1, 15.2, 15.3, 15.4_

- [ ] 4.6.1 Set up state management and constants
  - Create state object with signalParams (n_points, frequency, noise_std, add_trend, seed)

  - Add filterParams to state (moving_average: {enabled, window_size}, gaussian: {enabled, sigma})
  - Add currentData to state (initially null)
  - Define DEFAULT_STATE constant with initial values for easy reset
  - _Requirements: 13.2, 13.3_

- [ ] 4.6.2 Implement API communication functions
  - Write async function generateSignal() that POSTs to /api/generate_signal with state.signalParams

  - Parse JSON response and store in state.currentData
  - Handle fetch errors with try-catch and display user-friendly error message
  - Write async function applyFilters() that POSTs to /api/smooth with current signal and enabled filters
  - Return smoothed results and metrics
  - Handle fetch errors gracefully
  - _Requirements: 6.4, 15.3, 15.4_

- [ ] 4.6.3 Implement plot update function
  - Write function updatePlot(data, smoothedData) that creates Plotly traces
  - Create trace for noisy signal (scatter plot, gray color, small markers)
  - Create trace for clean signal if available (line plot, light color, dashed)

  - Create trace for moving average result if enabled (line plot, blue color)
  - Create trace for Gaussian result if enabled (line plot, purple color)
  - Configure layout with title, axis labels, legend, responsive sizing
  - Call Plotly.newPlot() to render in #plot div
  - _Requirements: 6.2, 8.4, 9.4, 15.1_

- [ ] 4.6.4 Implement metrics display function
  - Write function updateMetrics(metrics) that updates #metrics div
  - Create HTML table or list showing filter names and compute times
  - Display smoothness metric if computed (optional enhancement)


  - Format numbers with appropriate precision (e.g., 2 decimal places for ms)
  - Clear metrics if no filters are enabled
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 4.6.5 Implement event handlers and UI updates

  - Write setupEventListeners() function to attach all event handlers
  - Add input event listeners to all range inputs to update value display spans
  - Add change event listeners to range inputs with debouncing (200ms delay)
  - On parameter change, call applyFilters() and updatePlot()
  - Add click handler to "Generate New Signal" button to call generateSignal() then applyFilters()
  - Add click handler to "Download Plot" button to call Plotly.downloadImage()

  - Add click handler to "Reset" button to restore DEFAULT_STATE and regenerate
  - Add change listeners to filter enable/disable checkboxes
  - Show loading indicator during async operations (optional)
  - _Requirements: 6.3, 6.4, 7.4, 8.3, 9.3, 12.2, 12.3, 12.4, 12.5, 13.2, 13.3, 13.4, 13.5, 15.2_

- [ ] 4.6.6 Implement initialization function
  - Write init() function that calls setupEventListeners()
  - Call generateSignal() to create initial signal
  - Call applyFilters() to compute initial smoothed results

  - Call updatePlot() to render initial visualization
  - Call init() when DOM is loaded (DOMContentLoaded event or at end of script)
  - _Requirements: 6.1, 6.2_

- [ ] 5. Create Jupyter notebook demonstration
  - _Requirements: 4, 5, 17, 19_

- [x] 5.1 Create notebooks/demo.ipynb with introduction

  - Create new Jupyter notebook file
  - Add markdown cell with title "Signal Smoothing by Convolution"
  - Add markdown cell explaining the purpose: demonstrate moving average and Gaussian filters
  - Add markdown cell with calculus connection overview (convolution as integral, smoothing reduces derivatives)
  - _Requirements: 17.1, 19.2, 19.3, 19.4_

- [ ] 5.2 Add setup and imports cell
  - Create code cell with imports: numpy, matplotlib.pyplot
  - Import smoothing functions: from src.smoothing import moving_average, gaussian_smooth, gaussian_kernel
  - Set matplotlib style for better-looking plots: plt.style.use('seaborn-v0_8-darkgrid') or similar
  - Set random seed for reproducibility: np.random.seed(42)

  - _Requirements: 17.2, 4.3_

- [ ] 5.3 Add signal generation cell
  - Create code cell to generate synthetic noisy signal
  - Define parameters: n_points=2000, frequency=5.0, noise_std=0.6
  - Generate time array: t = np.linspace(0, 2, n_points)

  - Generate clean signal: clean = np.sin(2 * np.pi * frequency * t)
  - Generate noisy signal: noisy = clean + np.random.normal(0, noise_std, n_points)
  - Plot original signals using matplotlib (clean and noisy)
  - Add title, labels, legend, grid
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 17.2_

- [ ] 5.4 Add moving average demonstration cell
  - Create markdown cell explaining moving average filter and O(n) complexity
  - Create code cell applying moving average with different window sizes (e.g., 5, 11, 31)
  - Store results in variables: ma_5, ma_11, ma_31

  - Create subplot figure showing noisy signal and all moving average results
  - Add legend distinguishing different window sizes
  - Add markdown cell discussing effect of window size on smoothness vs lag
  - _Requirements: 17.3, 1.2_


- [ ] 5.5 Add Gaussian smoothing demonstration cell
  - Create markdown cell explaining Gaussian filter and kernel shape
  - Create code cell to visualize Gaussian kernel
  - Generate kernel with sigma=2.5: kernel = gaussian_kernel(2.5)
  - Plot kernel shape
  - Create code cell applying Gaussian smoothing with different sigmas (e.g., 1.0, 2.5, 5.0)
  - Store results: gauss_1, gauss_2_5, gauss_5
  - Create subplot figure showing noisy signal and all Gaussian results


  - Add legend distinguishing different sigma values
  - Add markdown cell discussing effect of sigma on smoothness
  - _Requirements: 17.3, 2.1, 2.2_

- [ ] 5.6 Add comparison and analysis cell
  - Create code cell comparing best moving average vs best Gaussian
  - Compute simple smoothness metric (e.g., standard deviation of second derivative)
  - Create side-by-side comparison plot
  - Add markdown cell discussing trade-offs between filter types
  - _Requirements: 17.4_

- [ ] 5.7 Add calculus connection cell
  - Create markdown cell with detailed calculus explanation
  - Explain discrete convolution as Riemann sum approximation of continuous convolution integral
  - Show mathematical formulas using LaTeX in markdown
  - Create code cell demonstrating derivative smoothing

  - Compute numerical derivative of noisy signal: np.diff(noisy)
  - Compute numerical derivative of smoothed signal: np.diff(smoothed)
  - Plot both derivatives to show smoothing effect
  - Add markdown cell explaining how smoothing reduces high-frequency components (high derivatives)
  - _Requirements: 19.2, 19.3, 19.4_

- [ ] 5.8 Add visualization saving cell
  - Create code cell to save final comparison plot
  - Use plt.savefig('plots/results.png', dpi=150, bbox_inches='tight')
  - Add markdown cell noting that plot has been saved
  - _Requirements: 5.4, 5.5_

- [ ] 5.9 Test notebook execution
  - Run all cells in sequence to verify no errors
  - Check that all plots render correctly
  - Verify plots/results.png is created
  - _Requirements: 17.5_

- [ ] 6. Create project documentation
  - _Requirements: 19, 16_

- [ ] 6.1 Write README.md
  - Add project title and brief description
  - Add "Overview" section explaining signal smoothing and convolution
  - Add "Calculus Connection" section with subsections:


    - Convolution as Integral (continuous vs discrete formulas)
    - Smoothing and Derivatives (how smoothing reduces high-order derivatives)
    - Riemann Sums (how discrete sums approximate integrals)
  - Add "Features" section listing: moving average filter, Gaussian filter, web interface, Jupyter notebook
  - Add "Installation" section with pip install -r requirements.txt
  - Add "Usage" section with two subsections:
    - Web Application: python app.py, then open http://localhost:5000
    - Jupyter Notebook: jupyter notebook notebooks/demo.ipynb
  - Add "Project Structure" section showing directory tree
  - Add "Dependencies" section listing main packages
  - Add "Testing" section with pytest command
  - Add "License" section (if applicable)
  - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 16.1, 16.2, 16.3, 16.5_

- [ ] 6.2 Add usage examples to README
  - Add Python code example showing how to use smoothing functions directly
  - Add example of generating signal and applying filters
  - Add example output or screenshot placeholder
  - _Requirements: 19.5_



- [ ] 7. Final integration and testing
  - _Requirements: 6.1, 15.5, 16.4, 16.5_

- [ ] 7.1 Test complete web application workflow
  - Start Flask server: python app.py
  - Open browser to http://localhost:5000
  - Verify page loads without console errors
  - Test generating signal with default parameters
  - Test adjusting frequency slider and verify plot updates
  - Test adjusting noise slider and verify plot updates
  - Test toggling trend checkbox
  - Test adjusting moving average window size slider
  - Test adjusting Gaussian sigma slider
  - Test disabling/enabling filters via checkboxes
  - Test download button produces valid PNG file
  - Test reset button restores all defaults
  - Verify metrics display shows compute times
  - Test in multiple browsers (Chrome, Firefox, Safari, Edge if available)
  - _Requirements: 6.1, 6.3, 6.4, 7.4, 8.3, 9.3, 12.1, 12.2, 13.1, 15.5_

- [ ] 7.2 Run all unit tests
  - Execute pytest from project root
  - Verify all tests pass
  - Check test coverage if coverage tool is available
  - Fix any failing tests
  - _Requirements: 18.5_

- [ ] 7.3 Verify notebook execution
  - Open notebooks/demo.ipynb in Jupyter
  - Run all cells from top to bottom
  - Verify no errors occur
  - Check that plots/results.png is created
  - Verify all plots render correctly
  - _Requirements: 17.5_

- [ ] 7.4 Create sample data file (optional)
  - Generate a sample noisy signal
  - Save to data/sample_signal.npy using np.save()
  - Add brief comment in README about sample data
  - _Requirements: 22.4_

- [ ] 7.5 Final review and cleanup
  - Review all code for consistent style and formatting
  - Ensure all functions have docstrings
  - Remove any debug print statements
  - Verify .gitignore is complete
  - Check that all requirements are met
  - Test installation in fresh virtual environment
  - _Requirements: 16.5, 20.1, 20.5, 21.1, 21.2, 21.3, 22.5_
