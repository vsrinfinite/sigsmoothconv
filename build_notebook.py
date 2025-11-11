"""Script to build the complete demo notebook."""
import json

# Load existing notebook
with open('notebooks/demo.ipynb', 'r') as f:
    nb = json.load(f)

# Add signal generation cell
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Generate synthetic noisy signal\n",
        "n_points = 2000\n",
        "frequency = 5.0\n",
        "noise_std = 0.6\n",
        "\n",
        "# Time array (2 seconds)\n",
        "t = np.linspace(0, 2, n_points)\n",
        "\n",
        "# Clean sinusoidal signal\n",
        "clean = np.sin(2 * np.pi * frequency * t)\n",
        "\n",
        "# Add Gaussian noise\n",
        "noisy = clean + np.random.normal(0, noise_std, n_points)\n",
        "\n",
        "# Plot original signals\n",
        "plt.figure(figsize=(14, 5))\n",
        "plt.plot(t, clean, 'g-', label='Clean Signal', linewidth=2, alpha=0.7)\n",
        "plt.plot(t, noisy, 'gray', label='Noisy Signal', linewidth=0.5, alpha=0.5)\n",
        "plt.xlabel('Time (seconds)')\n",
        "plt.ylabel('Amplitude')\n",
        "plt.title('Original Signal: Clean vs Noisy')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f'Generated signal with {n_points} points')\n",
        "print(f'Signal-to-noise ratio: {np.std(clean)/noise_std:.2f}')"
    ]
})

# Add moving average markdown
nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Moving Average Filter\n",
        "\n",
        "The moving average filter replaces each point with the average of surrounding points. Our implementation uses the cumulative sum trick for O(n) time complexity.\n",
        "\n",
        "**Algorithm**: For window size $w$:\n",
        "$$y[n] = \\frac{1}{w} \\sum_{k=-(w-1)/2}^{(w-1)/2} x[n+k]$$"
    ]
})

# Add moving average demo cell
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Apply moving average with different window sizes\n",
        "ma_5 = moving_average(noisy, window_size=5)\n",
        "ma_11 = moving_average(noisy, window_size=11)\n",
        "ma_31 = moving_average(noisy, window_size=31)\n",
        "\n",
        "# Plot comparison\n",
        "plt.figure(figsize=(14, 8))\n",
        "\n",
        "plt.subplot(2, 1, 1)\n",
        "plt.plot(t, noisy, 'gray', label='Noisy', linewidth=0.5, alpha=0.5)\n",
        "plt.plot(t, ma_5, 'b-', label='Window=5', linewidth=2)\n",
        "plt.plot(t, ma_11, 'r-', label='Window=11', linewidth=2)\n",
        "plt.plot(t, ma_31, 'g-', label='Window=31', linewidth=2)\n",
        "plt.xlabel('Time (seconds)')\n",
        "plt.ylabel('Amplitude')\n",
        "plt.title('Moving Average with Different Window Sizes')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "\n",
        "# Zoom in on a section\n",
        "plt.subplot(2, 1, 2)\n",
        "zoom_start, zoom_end = 500, 700\n",
        "plt.plot(t[zoom_start:zoom_end], noisy[zoom_start:zoom_end], 'gray', label='Noisy', linewidth=0.5, alpha=0.5)\n",
        "plt.plot(t[zoom_start:zoom_end], clean[zoom_start:zoom_end], 'k--', label='Clean', linewidth=1.5, alpha=0.7)\n",
        "plt.plot(t[zoom_start:zoom_end], ma_5[zoom_start:zoom_end], 'b-', label='Window=5', linewidth=2)\n",
        "plt.plot(t[zoom_start:zoom_end], ma_11[zoom_start:zoom_end], 'r-', label='Window=11', linewidth=2)\n",
        "plt.plot(t[zoom_start:zoom_end], ma_31[zoom_start:zoom_end], 'g-', label='Window=31', linewidth=2)\n",
        "plt.xlabel('Time (seconds)')\n",
        "plt.ylabel('Amplitude')\n",
        "plt.title('Zoomed View: Effect of Window Size')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('Larger windows produce smoother results but introduce more lag')"
    ]
})

# Add Gaussian filter markdown
nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Gaussian Filter\n",
        "\n",
        "The Gaussian filter uses a Gaussian-shaped kernel for weighted averaging. It provides smooth, continuous filtering with well-defined frequency response.\n",
        "\n",
        "**Kernel**: $G(x) = \\exp\\left(-\\frac{x^2}{2\\sigma^2}\\right)$\n",
        "\n",
        "The kernel is normalized so $\\sum G(x) = 1$."
    ]
})

# Add Gaussian kernel visualization
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize Gaussian kernels\n",
        "sigmas = [1.0, 2.5, 5.0]\n",
        "plt.figure(figsize=(12, 4))\n",
        "\n",
        "for sigma in sigmas:\n",
        "    kernel = gaussian_kernel(sigma)\n",
        "    x = np.arange(len(kernel)) - len(kernel)//2\n",
        "    plt.plot(x, kernel, 'o-', label=f'σ={sigma}', linewidth=2, markersize=4)\n",
        "\n",
        "plt.xlabel('Position')\n",
        "plt.ylabel('Weight')\n",
        "plt.title('Gaussian Kernels with Different σ Values')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('Larger σ produces wider, smoother kernels')"
    ]
})

# Add Gaussian smoothing demo
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Apply Gaussian smoothing with different sigmas\n",
        "gauss_1 = gaussian_smooth(noisy, sigma=1.0)\n",
        "gauss_2_5 = gaussian_smooth(noisy, sigma=2.5)\n",
        "gauss_5 = gaussian_smooth(noisy, sigma=5.0)\n",
        "\n",
        "# Plot comparison\n",
        "plt.figure(figsize=(14, 8))\n",
        "\n",
        "plt.subplot(2, 1, 1)\n",
        "plt.plot(t, noisy, 'gray', label='Noisy', linewidth=0.5, alpha=0.5)\n",
        "plt.plot(t, gauss_1, 'b-', label='σ=1.0', linewidth=2)\n",
        "plt.plot(t, gauss_2_5, 'r-', label='σ=2.5', linewidth=2)\n",
        "plt.plot(t, gauss_5, 'g-', label='σ=5.0', linewidth=2)\n",
        "plt.xlabel('Time (seconds)')\n",
        "plt.ylabel('Amplitude')\n",
        "plt.title('Gaussian Smoothing with Different σ Values')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "\n",
        "# Zoom in\n",
        "plt.subplot(2, 1, 2)\n",
        "plt.plot(t[zoom_start:zoom_end], noisy[zoom_start:zoom_end], 'gray', label='Noisy', linewidth=0.5, alpha=0.5)\n",
        "plt.plot(t[zoom_start:zoom_end], clean[zoom_start:zoom_end], 'k--', label='Clean', linewidth=1.5, alpha=0.7)\n",
        "plt.plot(t[zoom_start:zoom_end], gauss_1[zoom_start:zoom_end], 'b-', label='σ=1.0', linewidth=2)\n",
        "plt.plot(t[zoom_start:zoom_end], gauss_2_5[zoom_start:zoom_end], 'r-', label='σ=2.5', linewidth=2)\n",
        "plt.plot(t[zoom_start:zoom_end], gauss_5[zoom_start:zoom_end], 'g-', label='σ=5.0', linewidth=2)\n",
        "plt.xlabel('Time (seconds)')\n",
        "plt.ylabel('Amplitude')\n",
        "plt.title('Zoomed View: Effect of σ')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('Larger σ produces smoother results')"
    ]
})

# Add comparison cell
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Compare best moving average vs best Gaussian\n",
        "best_ma = moving_average(noisy, window_size=11)\n",
        "best_gauss = gaussian_smooth(noisy, sigma=2.5)\n",
        "\n",
        "# Compute smoothness metric (std of second derivative)\n",
        "def smoothness_metric(signal):\n",
        "    second_deriv = np.diff(signal, n=2)\n",
        "    return np.std(second_deriv)\n",
        "\n",
        "ma_smoothness = smoothness_metric(best_ma)\n",
        "gauss_smoothness = smoothness_metric(best_gauss)\n",
        "\n",
        "# Side-by-side comparison\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "axes[0].plot(t, noisy, 'gray', label='Noisy', linewidth=0.5, alpha=0.5)\n",
        "axes[0].plot(t, clean, 'k--', label='Clean', linewidth=1.5, alpha=0.7)\n",
        "axes[0].plot(t, best_ma, 'b-', label='Moving Average (w=11)', linewidth=2)\n",
        "axes[0].set_xlabel('Time (seconds)')\n",
        "axes[0].set_ylabel('Amplitude')\n",
        "axes[0].set_title(f'Moving Average\\nSmoothness: {ma_smoothness:.4f}')\n",
        "axes[0].legend()\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "axes[1].plot(t, noisy, 'gray', label='Noisy', linewidth=0.5, alpha=0.5)\n",
        "axes[1].plot(t, clean, 'k--', label='Clean', linewidth=1.5, alpha=0.7)\n",
        "axes[1].plot(t, best_gauss, 'r-', label='Gaussian (σ=2.5)', linewidth=2)\n",
        "axes[1].set_xlabel('Time (seconds)')\n",
        "axes[1].set_ylabel('Amplitude')\n",
        "axes[1].set_title(f'Gaussian Filter\\nSmoothness: {gauss_smoothness:.4f}')\n",
        "axes[1].legend()\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('\\nComparison:')\n",
        "print(f'Moving Average smoothness: {ma_smoothness:.4f}')\n",
        "print(f'Gaussian smoothness: {gauss_smoothness:.4f}')\n",
        "print('\\nLower smoothness metric = smoother signal')"
    ]
})

# Add calculus connection markdown
nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Calculus Connections\n",
        "\n",
        "### Discrete Convolution as Riemann Sum\n",
        "\n",
        "The discrete convolution sum approximates the continuous convolution integral:\n",
        "\n",
        "$$\\sum_{k} f[k]g[n-k] \\Delta t \\approx \\int f(\\tau)g(t-\\tau) d\\tau$$\n",
        "\n",
        "This is a Riemann sum approximation where $\\Delta t$ is the sampling interval.\n",
        "\n",
        "### Smoothing Reduces High-Order Derivatives\n",
        "\n",
        "Let's demonstrate how smoothing reduces the magnitude of derivatives."
    ]
})

# Add derivative demonstration
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Compute numerical derivatives\n",
        "dt = t[1] - t[0]\n",
        "\n",
        "# First derivative (velocity)\n",
        "noisy_deriv = np.diff(noisy) / dt\n",
        "smoothed_deriv = np.diff(best_gauss) / dt\n",
        "\n",
        "# Second derivative (acceleration)\n",
        "noisy_deriv2 = np.diff(noisy, n=2) / (dt**2)\n",
        "smoothed_deriv2 = np.diff(best_gauss, n=2) / (dt**2)\n",
        "\n",
        "# Plot derivatives\n",
        "fig, axes = plt.subplots(3, 1, figsize=(14, 10))\n",
        "\n",
        "# Original signals\n",
        "axes[0].plot(t, noisy, 'gray', label='Noisy', linewidth=0.5, alpha=0.5)\n",
        "axes[0].plot(t, best_gauss, 'r-', label='Smoothed', linewidth=2)\n",
        "axes[0].set_ylabel('Signal')\n",
        "axes[0].set_title('Original Signals')\n",
        "axes[0].legend()\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "# First derivatives\n",
        "axes[1].plot(t[:-1], noisy_deriv, 'gray', label='Noisy', linewidth=0.5, alpha=0.5)\n",
        "axes[1].plot(t[:-1], smoothed_deriv, 'r-', label='Smoothed', linewidth=2)\n",
        "axes[1].set_ylabel('First Derivative')\n",
        "axes[1].set_title('First Derivatives (Rate of Change)')\n",
        "axes[1].legend()\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "\n",
        "# Second derivatives\n",
        "axes[2].plot(t[:-2], noisy_deriv2, 'gray', label='Noisy', linewidth=0.5, alpha=0.5)\n",
        "axes[2].plot(t[:-2], smoothed_deriv2, 'r-', label='Smoothed', linewidth=2)\n",
        "axes[2].set_xlabel('Time (seconds)')\n",
        "axes[2].set_ylabel('Second Derivative')\n",
        "axes[2].set_title('Second Derivatives (Curvature)')\n",
        "axes[2].legend()\n",
        "axes[2].grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('Smoothing dramatically reduces high-frequency components (high derivatives)')\n",
        "print(f'Noisy 2nd derivative std: {np.std(noisy_deriv2):.2f}')\n",
        "print(f'Smoothed 2nd derivative std: {np.std(smoothed_deriv2):.2f}')\n",
        "print(f'Reduction factor: {np.std(noisy_deriv2)/np.std(smoothed_deriv2):.1f}x')"
    ]
})

# Add save plot cell
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Save final comparison plot\n",
        "plt.figure(figsize=(14, 6))\n",
        "plt.plot(t, noisy, 'gray', label='Noisy Signal', linewidth=0.5, alpha=0.5)\n",
        "plt.plot(t, clean, 'k--', label='Clean Signal', linewidth=1.5, alpha=0.7)\n",
        "plt.plot(t, best_ma, 'b-', label='Moving Average (w=11)', linewidth=2)\n",
        "plt.plot(t, best_gauss, 'r-', label='Gaussian (σ=2.5)', linewidth=2)\n",
        "plt.xlabel('Time (seconds)', fontsize=12)\n",
        "plt.ylabel('Amplitude', fontsize=12)\n",
        "plt.title('Signal Smoothing Comparison', fontsize=14, fontweight='bold')\n",
        "plt.legend(fontsize=11)\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../plots/results.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print('Plot saved to plots/results.png')"
    ]
})

# Add conclusion markdown
nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Conclusion\n",
        "\n",
        "We've demonstrated:\n",
        "\n",
        "1. **Moving Average**: Fast O(n) smoothing with simple averaging\n",
        "2. **Gaussian Filter**: Smooth, continuous filtering with controlled frequency response\n",
        "3. **Calculus Connection**: Discrete operations approximate continuous integrals and derivatives\n",
        "\n",
        "### Key Takeaways\n",
        "\n",
        "- Larger window sizes / sigma values produce more smoothing\n",
        "- Smoothing reduces high-frequency noise (high derivatives)\n",
        "- Discrete convolution approximates continuous integral convolution\n",
        "- Trade-off between noise reduction and signal distortion\n",
        "\n",
        "### Next Steps\n",
        "\n",
        "Try the interactive web application to experiment with different parameters in real-time!"
    ]
})

# Save the updated notebook
with open('notebooks/demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Notebook complete with {len(nb["cells"])} cells')
