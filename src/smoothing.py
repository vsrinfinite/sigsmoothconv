"""
Signal smoothing functions using convolution-based filters.

This module implements moving average and Gaussian smoothing filters
for one-dimensional signals, with efficient O(n) algorithms and proper
edge handling using reflection padding.
"""

import numpy as np


def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute centered moving average using cumulative sum algorithm.
    
    This function applies a moving average filter to smooth a 1D signal.
    The implementation uses the cumulative sum trick for O(n) time complexity,
    making it efficient even for large signals and window sizes.
    
    Args:
        signal: 1D numpy array of signal values
        window_size: Positive odd integer specifying the window size.
                    Must be odd to ensure centered averaging.
        
    Returns:
        Smoothed signal as 1D numpy array with the same length as input.
        All values are in float64 precision.
        
    Raises:
        ValueError: If signal is not 1D
        ValueError: If window_size is not a positive odd integer
        ValueError: If window_size is larger than signal length
        ValueError: If signal contains NaN or infinite values
        
    Algorithm:
        1. Convert input to float64 for numerical stability
        2. Apply reflection padding: (window_size-1)//2 on each side
        3. Compute cumulative sum with prepended zero
        4. Extract windowed averages: (cumsum[w:] - cumsum[:-w]) / w
        5. Trim result to original signal length
        
    Time Complexity: O(n) where n is the signal length
    Space Complexity: O(n) for padded array
    
    Example:
        >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> smoothed = moving_average(signal, window_size=3)
        >>> # Result: approximately [1.33, 2.0, 3.0, 4.0, 4.67]
    """
    # Input validation
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D array")
    
    if not isinstance(window_size, (int, np.integer)):
        raise ValueError("Window size must be an integer")
    
    if window_size < 1:
        raise ValueError("Window size must be positive")
    
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd for centered averaging")
    
    if len(signal) < window_size:
        raise ValueError(
            f"Signal length ({len(signal)}) must be at least window_size ({window_size})"
        )
    
    # Convert to float64 for numerical precision
    signal = signal.astype(np.float64)
    
    # Check for NaN or infinite values
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values")
    
    # Compute padding amounts for centered window
    pad_left = (window_size - 1) // 2
    pad_right = window_size - 1 - pad_left
    
    # Apply reflection padding
    padded_signal = np.pad(signal, (pad_left, pad_right), mode='reflect')
    
    # Compute cumulative sum with prepended zero
    cumsum = np.concatenate(([0.0], np.cumsum(padded_signal)))
    
    # Extract windowed averages using cumsum trick
    windowed_sum = cumsum[window_size:] - cumsum[:-window_size]
    result = windowed_sum / window_size
    
    # Trim to original signal length
    return result[:len(signal)]



def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """
    Generate normalized 1D Gaussian kernel.
    
    Creates a discrete Gaussian kernel suitable for convolution-based smoothing.
    The kernel is normalized so that the sum of all values equals 1.0, ensuring
    that the smoothed signal maintains the same overall scale as the input.
    
    Args:
        sigma: Standard deviation of the Gaussian distribution (must be > 0).
               Larger sigma produces wider, smoother kernels.
        truncate: Number of standard deviations to include in the kernel
                 (default 4.0). The kernel extends from -truncate*sigma to
                 +truncate*sigma. Default of 4.0 captures ~99.99% of the
                 Gaussian mass.
        
    Returns:
        Normalized 1D numpy array where sum equals 1.0 within tolerance of 1e-12.
        The kernel is symmetric and centered at index len(kernel)//2.
        
    Raises:
        ValueError: If sigma <= 0
        ValueError: If truncate <= 0
        
    Algorithm:
        1. Compute radius = ceil(truncate * sigma)
        2. Create x values from -radius to +radius
        3. Compute Gaussian: exp(-x²/(2σ²))
        4. Normalize by dividing by sum
        5. Ensure minimum kernel length of 3 for numerical stability
        
    Mathematical Formula:
        G(x) = exp(-x²/(2σ²))
        Normalized: K(x) = G(x) / Σ G(x)
        
    Example:
        >>> kernel = gaussian_kernel(sigma=1.0)
        >>> len(kernel)  # With truncate=4.0, radius=4, length=9
        9
        >>> abs(kernel.sum() - 1.0) < 1e-12
        True
    """
    # Input validation
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    if truncate <= 0:
        raise ValueError("Truncate parameter must be positive")
    
    # Compute kernel radius
    radius = int(np.ceil(truncate * sigma))
    
    # Ensure minimum kernel length of 3 for very small sigma
    if radius < 1:
        radius = 1
    
    # Create x array from -radius to +radius
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    
    # Compute unnormalized Gaussian values
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    
    # Normalize so sum equals 1.0
    kernel = kernel / kernel.sum()
    
    return kernel



def gaussian_smooth(signal: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 1D signal.
    
    This function smooths a signal by convolving it with a Gaussian kernel.
    Gaussian smoothing is optimal for removing Gaussian noise and provides
    smooth, continuous filtering with well-defined frequency response.
    
    Args:
        signal: 1D numpy array of signal values
        sigma: Standard deviation of the Gaussian kernel (must be > 0).
              Larger sigma produces more smoothing.
        truncate: Kernel truncation parameter (default 4.0).
                 Passed to gaussian_kernel().
        
    Returns:
        Smoothed signal as 1D numpy array with the same length as input.
        All values are in float64 precision.
        
    Raises:
        ValueError: If signal is not 1D
        ValueError: If sigma <= 0
        ValueError: If signal contains NaN or infinite values
        
    Algorithm:
        1. Convert input to float64
        2. Generate Gaussian kernel using gaussian_kernel()
        3. Apply reflection padding based on kernel length
        4. Convolve padded signal with kernel
        5. Return result trimmed to original length
        
    Edge Handling:
        Uses reflection padding to minimize edge artifacts. The signal is
        extended by reflecting values across the boundaries before convolution.
        
    Time Complexity: O(n × m) where n is signal length, m is kernel length
    Space Complexity: O(n + m)
    
    Example:
        >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> smoothed = gaussian_smooth(signal, sigma=1.0)
        >>> # Result will be smoothly varying values close to input
    """
    # Input validation
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D array")
    
    # Convert to float64 for numerical precision
    signal = signal.astype(np.float64)
    
    # Check for NaN or infinite values
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values")
    
    # Generate Gaussian kernel
    kernel = gaussian_kernel(sigma, truncate)
    
    # Compute padding width (half kernel length on each side)
    pad_width = len(kernel) // 2
    
    # Apply reflection padding
    padded_signal = np.pad(signal, pad_width, mode='reflect')
    
    # Perform convolution
    # Using mode='valid' returns only the part where kernel fully overlaps
    # Due to our padding, this gives us exactly the original signal length
    result = np.convolve(padded_signal, kernel, mode='valid')
    
    return result
