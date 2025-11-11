"""
Integration tests for Flask API endpoints.

Tests verify that the web API correctly handles requests, validates parameters,
and returns properly formatted responses.
"""

import pytest
import json
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestGenerateSignalEndpoint:
    """Tests for /api/generate_signal endpoint."""
    
    def test_generate_signal_default_params(self, client):
        """Test signal generation with default parameters."""
        response = client.post('/api/generate_signal',
                              json={},
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert 'time' in data
        assert 'signal' in data
        assert 'clean' in data
        
        # Check array lengths
        assert len(data['time']) == 2000  # default n_points
        assert len(data['signal']) == 2000
        assert len(data['clean']) == 2000
    
    def test_generate_signal_custom_params(self, client):
        """Test signal generation with custom parameters."""
        response = client.post('/api/generate_signal',
                              json={
                                  'n_points': 1000,
                                  'frequency': 10.0,
                                  'noise_std': 0.3,
                                  'add_trend': False,
                                  'seed': 42
                              },
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check custom length
        assert len(data['time']) == 1000
        assert len(data['signal']) == 1000
        assert len(data['clean']) == 1000
    
    def test_generate_signal_with_trend(self, client):
        """Test signal generation with trend enabled."""
        response = client.post('/api/generate_signal',
                              json={
                                  'n_points': 500,
                                  'add_trend': True,
                                  'seed': 42
                              },
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Signal should have trend (check clean signal has increasing trend)
        clean = data['clean']
        # Average of last 50 points should be higher than average of first 50 points
        assert sum(clean[-50:]) / 50 > sum(clean[:50]) / 50
    
    def test_generate_signal_invalid_n_points_too_small(self, client):
        """Test error handling for n_points too small."""
        response = client.post('/api/generate_signal',
                              json={'n_points': 50},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['code'] == 'INVALID_PARAMETER'
    
    def test_generate_signal_invalid_n_points_too_large(self, client):
        """Test error handling for n_points too large."""
        response = client.post('/api/generate_signal',
                              json={'n_points': 20000},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_generate_signal_invalid_frequency(self, client):
        """Test error handling for invalid frequency."""
        response = client.post('/api/generate_signal',
                              json={'frequency': 100.0},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_generate_signal_invalid_noise_std(self, client):
        """Test error handling for invalid noise_std."""
        response = client.post('/api/generate_signal',
                              json={'noise_std': -0.5},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data


class TestSmoothEndpoint:
    """Tests for /api/smooth endpoint."""
    
    def test_smooth_moving_average_only(self, client):
        """Test smoothing with only moving average filter."""
        # Generate a simple signal
        signal = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0]
        
        response = client.post('/api/smooth',
                              json={
                                  'signal': signal,
                                  'filters': {
                                      'moving_average': {'window_size': 3}
                                  }
                              },
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert 'moving_average' in data
        assert 'result' in data['moving_average']
        assert 'compute_time_ms' in data['moving_average']
        
        # Check result length
        assert len(data['moving_average']['result']) == len(signal)
        
        # Check compute time is positive
        assert data['moving_average']['compute_time_ms'] > 0
    
    def test_smooth_gaussian_only(self, client):
        """Test smoothing with only Gaussian filter."""
        signal = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0]
        
        response = client.post('/api/smooth',
                              json={
                                  'signal': signal,
                                  'filters': {
                                      'gaussian': {'sigma': 1.0}
                                  }
                              },
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert 'gaussian' in data
        assert 'result' in data['gaussian']
        assert 'compute_time_ms' in data['gaussian']
        
        # Check result length
        assert len(data['gaussian']['result']) == len(signal)
        
        # Check compute time is positive
        assert data['gaussian']['compute_time_ms'] > 0
    
    def test_smooth_both_filters(self, client):
        """Test smoothing with both filters enabled."""
        signal = list(range(20))
        
        response = client.post('/api/smooth',
                              json={
                                  'signal': signal,
                                  'filters': {
                                      'moving_average': {'window_size': 5},
                                      'gaussian': {'sigma': 2.0}
                                  }
                              },
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Both filters should be in response
        assert 'moving_average' in data
        assert 'gaussian' in data
        
        # Both should have results
        assert len(data['moving_average']['result']) == len(signal)
        assert len(data['gaussian']['result']) == len(signal)
    
    def test_smooth_invalid_window_size_even(self, client):
        """Test error handling for even window size."""
        signal = list(range(20))
        
        response = client.post('/api/smooth',
                              json={
                                  'signal': signal,
                                  'filters': {
                                      'moving_average': {'window_size': 4}
                                  }
                              },
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'odd' in data['message'].lower()
    
    def test_smooth_invalid_window_size_too_large(self, client):
        """Test error handling for window size too large."""
        signal = list(range(20))
        
        response = client.post('/api/smooth',
                              json={
                                  'signal': signal,
                                  'filters': {
                                      'moving_average': {'window_size': 201}
                                  }
                              },
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_smooth_invalid_sigma_too_small(self, client):
        """Test error handling for sigma too small."""
        signal = list(range(20))
        
        response = client.post('/api/smooth',
                              json={
                                  'signal': signal,
                                  'filters': {
                                      'gaussian': {'sigma': 0.1}
                                  }
                              },
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_smooth_invalid_sigma_too_large(self, client):
        """Test error handling for sigma too large."""
        signal = list(range(20))
        
        response = client.post('/api/smooth',
                              json={
                                  'signal': signal,
                                  'filters': {
                                      'gaussian': {'sigma': 15.0}
                                  }
                              },
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_smooth_signal_too_short(self, client):
        """Test error handling for signal too short."""
        signal = [1.0, 2.0, 3.0]
        
        response = client.post('/api/smooth',
                              json={
                                  'signal': signal,
                                  'filters': {
                                      'moving_average': {'window_size': 3}
                                  }
                              },
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_smooth_compute_time_present(self, client):
        """Test that compute_time_ms is always present and positive."""
        signal = list(range(100))
        
        response = client.post('/api/smooth',
                              json={
                                  'signal': signal,
                                  'filters': {
                                      'moving_average': {'window_size': 11},
                                      'gaussian': {'sigma': 2.5}
                                  }
                              },
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check compute times are present and positive
        assert data['moving_average']['compute_time_ms'] > 0
        assert data['gaussian']['compute_time_ms'] > 0
        
        # Check they are numbers (not strings)
        assert isinstance(data['moving_average']['compute_time_ms'], (int, float))
        assert isinstance(data['gaussian']['compute_time_ms'], (int, float))
