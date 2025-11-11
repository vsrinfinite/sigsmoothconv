"""
Flask web application for interactive signal smoothing visualization.

This application provides a web interface for experimenting with moving average
and Gaussian smoothing filters in real-time. Users can adjust signal parameters
and filter settings through an interactive UI.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
from src.smoothing import moving_average, gaussian_smooth

# Create Flask application
app = Flask(__name__)

# Enable CORS for cross-origin requests
CORS(app)


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/generate_signal', methods=['POST'])
def generate_signal():
    """
    Generate a synthetic noisy signal.
    
    Expected JSON request body:
    {
        "n_points": 2000,
        "frequency": 5.0,
        "noise_std": 0.6,
        "add_trend": false,
        "seed": 42
    }
    
    Returns JSON response:
    {
        "time": [0.0, 0.001, ...],
        "signal": [noisy values...],
        "clean": [clean values...]
    }
    """
    try:
        # Parse request parameters
        data = request.get_json()
        
        n_points = data.get('n_points', 2000)
        frequency = data.get('frequency', 5.0)
        noise_std = data.get('noise_std', 0.6)
        add_trend = data.get('add_trend', False)
        seed = data.get('seed', None)
        
        # Validate parameters
        if n_points < 100 or n_points > 10000:
            return jsonify({
                'error': 'Invalid parameter',
                'message': 'n_points must be between 100 and 10000',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        if frequency < 0.1 or frequency > 50:
            return jsonify({
                'error': 'Invalid parameter',
                'message': 'frequency must be between 0.1 and 50 Hz',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        if noise_std < 0 or noise_std > 5.0:
            return jsonify({
                'error': 'Invalid parameter',
                'message': 'noise_std must be between 0 and 5.0',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate time array (2 seconds duration)
        duration = 2.0
        time = np.linspace(0, duration, n_points)
        
        # Generate clean sinusoidal signal
        clean = np.sin(2 * np.pi * frequency * time)
        
        # Add Gaussian noise
        noisy = clean + np.random.normal(0, noise_std, n_points)
        
        # Optionally add linear trend
        if add_trend:
            trend = np.linspace(0, 0.5, n_points)
            noisy = noisy + trend
            clean = clean + trend
        
        # Return JSON response
        return jsonify({
            'time': time.tolist(),
            'signal': noisy.tolist(),
            'clean': clean.tolist()
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e),
            'code': 'SERVER_ERROR'
        }), 500


@app.route('/api/smooth', methods=['POST'])
def smooth_signal():
    """
    Apply smoothing filters to a signal.
    
    Expected JSON request body:
    {
        "signal": [array of values...],
        "filters": {
            "moving_average": {"window_size": 11},
            "gaussian": {"sigma": 2.5}
        }
    }
    
    Returns JSON response:
    {
        "moving_average": {
            "result": [smoothed values...],
            "compute_time_ms": 1.2
        },
        "gaussian": {
            "result": [smoothed values...],
            "compute_time_ms": 2.5
        }
    }
    """
    try:
        import time
        
        # Parse request
        data = request.get_json()
        
        signal_list = data.get('signal', [])
        filters = data.get('filters', {})
        
        # Convert signal to numpy array
        signal = np.array(signal_list, dtype=np.float64)
        
        # Validate signal
        if len(signal) < 10:
            return jsonify({
                'error': 'Invalid parameter',
                'message': 'Signal must have at least 10 points',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        # Initialize results dictionary
        results = {}
        
        # Apply moving average filter if requested
        if 'moving_average' in filters:
            ma_params = filters['moving_average']
            window_size = ma_params.get('window_size', 11)
            
            # Validate window size
            if window_size < 3 or window_size > 101:
                return jsonify({
                    'error': 'Invalid parameter',
                    'message': 'Window size must be between 3 and 101',
                    'code': 'INVALID_PARAMETER'
                }), 400
            
            if window_size % 2 == 0:
                return jsonify({
                    'error': 'Invalid parameter',
                    'message': 'Window size must be odd',
                    'code': 'INVALID_PARAMETER'
                }), 400
            
            # Apply filter and measure time
            start_time = time.perf_counter()
            ma_result = moving_average(signal, window_size)
            end_time = time.perf_counter()
            
            compute_time_ms = (end_time - start_time) * 1000
            
            results['moving_average'] = {
                'result': ma_result.tolist(),
                'compute_time_ms': round(compute_time_ms, 3)
            }
        
        # Apply Gaussian filter if requested
        if 'gaussian' in filters:
            gauss_params = filters['gaussian']
            sigma = gauss_params.get('sigma', 2.5)
            
            # Validate sigma
            if sigma < 0.5 or sigma > 10.0:
                return jsonify({
                    'error': 'Invalid parameter',
                    'message': 'Sigma must be between 0.5 and 10.0',
                    'code': 'INVALID_PARAMETER'
                }), 400
            
            # Apply filter and measure time
            start_time = time.perf_counter()
            gauss_result = gaussian_smooth(signal, sigma)
            end_time = time.perf_counter()
            
            compute_time_ms = (end_time - start_time) * 1000
            
            results['gaussian'] = {
                'result': gauss_result.tolist(),
                'compute_time_ms': round(compute_time_ms, 3)
            }
        
        return jsonify(results)
    
    except ValueError as e:
        return jsonify({
            'error': 'Invalid parameter',
            'message': str(e),
            'code': 'INVALID_PARAMETER'
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e),
            'code': 'SERVER_ERROR'
        }), 500


if __name__ == '__main__':
    # Run the application on port 5000 with debug mode enabled
    app.run(host='0.0.0.0', port=5000, debug=True)
