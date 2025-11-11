"""
Unit tests for signal smoothing functions.

Tests verify correctness, edge cases, and numerical precision of
moving_average, gaussian_kernel, and gaussian_smooth functions.
"""

import numpy as np
import pytest
from src.smoothing import moving_average, gaussian_kernel, gaussian_smooth


class TestMovingAverage:
    """Tests for moving_average function."""
    
    def test_constant_signal(self):
        """Moving average of constant signal should return constant."""
        signal = np.ones(100)
        result = moving_average(signal, window_size=11)
        
        # All values should be 1.0 within tolerance
        assert np.allclose(result, 1.0, atol=1e-12)
    
    def test_output_length(self):
        """Output should have same length as input."""
        signal = np.random.randn(50)
        window_sizes = [3, 5, 11, 21]
        
        for window_size in window_sizes:
            result = moving_average(signal, window_size)
            assert len(result) == len(signal)
    
    def test_known_simple_case(self):
        """Test with known simple input."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(signal, window_size=3)
        
        # Expected: [1.33..., 2.0, 3.0, 4.0, 4.66...]
        # Middle values should be exact
        assert np.isclose(result[1], 2.0, atol=1e-10)
        assert np.isclose(result[2], 3.0, atol=1e-10)
        assert np.isclose(result[3], 4.0, atol=1e-10)
    
    def test_invalid_window_even(self):
        """Even window size should raise ValueError."""
        signal = np.ones(10)
        with pytest.raises(ValueError, match="odd"):
            moving_average(signal, window_size=4)
    
    def test_invalid_window_negative(self):
        """Negative window size should raise ValueError."""
        signal = np.ones(10)
        with pytest.raises(ValueError, match="positive"):
            moving_average(signal, window_size=-1)
    
    def test_invalid_window_zero(self):
        """Zero window size should raise ValueError."""
        signal = np.ones(10)
        with pytest.raises(ValueError, match="positive"):
            moving_average(signal, window_size=0)
    
    def test_window_larger_than_signal(self):
        """Window larger than signal should raise ValueError."""
        signal = np.ones(5)
        with pytest.raises(ValueError, match="at least"):
            moving_average(signal, window_size=11)
    
    def test_non_1d_signal(self):
        """Non-1D signal should raise ValueError."""
        signal = np.ones((5, 5))
        with pytest.raises(ValueError, match="1D"):
            moving_average(signal, window_size=3)
    
    def test_signal_with_nan(self):
        """Signal with NaN should raise ValueError."""
        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        with pytest.raises(ValueError, match="NaN"):
            moving_average(signal, window_size=3)
    
    def test_signal_with_inf(self):
        """Signal with infinite values should raise ValueError."""
        signal = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        with pytest.raises(ValueError, match="infinite"):
            moving_average(signal, window_size=3)
    
    def test_float64_precision(self):
        """Result should be in float64 precision."""
        signal = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = moving_average(signal, window_size=3)
        assert result.dtype == np.float64


class TestGaussianKernel:
    """Tests for gaussian_kernel function."""
    
    def test_normalization(self):
        """Gaussian kernel should sum to 1.0."""
        sigmas = [0.5, 1.0, 2.5, 5.0]
        
        for sigma in sigmas:
            kernel = gaussian_kernel(sigma)
            assert np.abs(kernel.sum() - 1.0) < 1e-12
    
    def test_symmetry(self):
        """Gaussian kernel should be symmetric."""
        kernel = gaussian_kernel(sigma=2.0)
        
        # Kernel should be symmetric around center
        assert np.allclose(kernel, kernel[::-1], atol=1e-12)
    
    def test_different_truncate(self):
        """Different truncate values should produce different kernel lengths."""
        sigma = 2.0
        kernel_3 = gaussian_kernel(sigma, truncate=3.0)
        kernel_4 = gaussian_kernel(sigma, truncate=4.0)
        kernel_5 = gaussian_kernel(sigma, truncate=5.0)
        
        # Larger truncate should give longer kernel
        assert len(kernel_5) > len(kernel_4) > len(kernel_3)
    
    def test_invalid_sigma_negative(self):
        """Negative sigma should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            gaussian_kernel(sigma=-1.0)
    
    def test_invalid_sigma_zero(self):
        """Zero sigma should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            gaussian_kernel(sigma=0.0)
    
    def test_invalid_truncate_negative(self):
        """Negative truncate should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            gaussian_kernel(sigma=1.0, truncate=-1.0)
    
    def test_small_sigma_minimum_length(self):
        """Very small sigma should still produce kernel of at least length 3."""
        kernel = gaussian_kernel(sigma=0.1)
        assert len(kernel) >= 3
    
    def test_kernel_peak_at_center(self):
        """Kernel should have maximum value at center."""
        kernel = gaussian_kernel(sigma=2.0)
        center_idx = len(kernel) // 2
        
        # Center should be the maximum value
        assert kernel[center_idx] == np.max(kernel)


class TestGaussianSmooth:
    """Tests for gaussian_smooth function."""
    
    def test_output_length(self):
        """Output should have same length as input."""
        signal = np.random.randn(100)
        sigmas = [0.5, 1.0, 2.5, 5.0]
        
        for sigma in sigmas:
            result = gaussian_smooth(signal, sigma)
            assert len(result) == len(signal)
    
    def test_constant_signal(self):
        """Gaussian smooth of constant signal should return constant."""
        signal = np.ones(100) * 5.0
        result = gaussian_smooth(signal, sigma=2.0)
        
        # All values should be 5.0 within tolerance
        assert np.allclose(result, 5.0, atol=1e-10)
    
    def test_smoothing_reduces_variance(self):
        """Smoothing should reduce signal variance."""
        np.random.seed(42)
        signal = np.random.randn(200)
        smoothed = gaussian_smooth(signal, sigma=2.0)
        
        # Smoothed signal should have lower variance
        assert np.var(smoothed) < np.var(signal)
    
    def test_invalid_sigma(self):
        """Invalid sigma should raise ValueError."""
        signal = np.ones(10)
        with pytest.raises(ValueError, match="positive"):
            gaussian_smooth(signal, sigma=-1.0)
    
    def test_non_1d_signal(self):
        """Non-1D signal should raise ValueError."""
        signal = np.ones((5, 5))
        with pytest.raises(ValueError, match="1D"):
            gaussian_smooth(signal, sigma=1.0)
    
    def test_signal_with_nan(self):
        """Signal with NaN should raise ValueError."""
        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        with pytest.raises(ValueError, match="NaN"):
            gaussian_smooth(signal, sigma=1.0)
    
    def test_float64_precision(self):
        """Result should be in float64 precision."""
        signal = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = gaussian_smooth(signal, sigma=1.0)
        assert result.dtype == np.float64
    
    def test_larger_sigma_more_smoothing(self):
        """Larger sigma should produce more smoothing."""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.5 * np.random.randn(100)
        
        smooth_1 = gaussian_smooth(signal, sigma=1.0)
        smooth_3 = gaussian_smooth(signal, sigma=3.0)
        
        # Larger sigma should have lower variance
        assert np.var(smooth_3) < np.var(smooth_1)


class TestNumericalAccuracy:
    """Tests for numerical precision and accuracy."""
    
    def test_moving_average_precision(self):
        """Moving average should maintain float64 precision."""
        signal = np.linspace(0, 10, 100)
        result = moving_average(signal, window_size=11)
        
        # Check dtype
        assert result.dtype == np.float64
        
        # Result should be close to linear signal (smoothing linear is identity-like)
        # Middle section should be very close to original
        assert np.allclose(result[20:80], signal[20:80], rtol=0.1)
    
    def test_gaussian_kernel_sum_precision(self):
        """Gaussian kernel sum should be exactly 1.0 within tight tolerance."""
        for sigma in [0.5, 1.0, 2.0, 5.0, 10.0]:
            kernel = gaussian_kernel(sigma)
            error = np.abs(kernel.sum() - 1.0)
            assert error < 1e-12, f"Kernel sum error {error} exceeds tolerance for sigma={sigma}"
    
    def test_known_analytical_result(self):
        """Test against known analytical result for simple case."""
        # For a constant signal, any smoothing should return the constant
        constant = 3.14159
        signal = np.full(50, constant)
        
        ma_result = moving_average(signal, window_size=7)
        gauss_result = gaussian_smooth(signal, sigma=2.0)
        
        # Both should return the constant within tight tolerance
        assert np.allclose(ma_result, constant, atol=1e-10)
        assert np.allclose(gauss_result, constant, atol=1e-10)
