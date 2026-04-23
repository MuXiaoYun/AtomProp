import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import warnings

# File path for the data
filepath = "trained_models/finetune_0127_nopre_bace/test_predictions_nopretrain.csv"

# Read file
with open(filepath, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

# Parse data
values = []
labels = []
n = 0

while 3*n + 2 < len(lines):
    # Get value from line 3n+1 (0-based indexing)
    try:
        val = float(lines[3*n + 1])
        label = int(lines[3*n + 2])
        
        # Only accept valid labels (0 or 1)
        if label in [0, 1]:
            values.append(val)
            labels.append(label)
    except (ValueError, IndexError):
        # Skip malformed lines
        pass
    n += 1

# Convert to numpy arrays
values = np.array(values)
labels = np.array(labels)

# Separate positive and negative samples
neg_values = values[labels == 0]
pos_values = values[labels == 1]

# Check if we have data
if len(neg_values) == 0 or len(pos_values) == 0:
    print("Warning: Missing either negative or positive samples")
    exit()

# Create figure
plt.figure(figsize=(12, 7))

# Determine optimal number of bins - using more bins for finer granularity
data_min = min(neg_values.min(), pos_values.min())
data_max = max(neg_values.max(), pos_values.max())
data_range = data_max - data_min

# Calculate number of bins using Scott's rule but with increased number
# Original: 3.5 * std / n^(1/3), we'll use smaller factor for more bins
neg_std = np.std(neg_values)
pos_std = np.std(pos_values)
avg_std = (neg_std + pos_std) / 2

# Calculate bins using Scott's rule with adjustment for more bins
neg_bin_width = 2.5 * neg_std / (len(neg_values) ** (1/3))  # Reduced from 3.5 to 2.5
pos_bin_width = 2.5 * pos_std / (len(pos_values) ** (1/3))  # Reduced from 3.5 to 2.5

# Use smaller bin width for more bins
bin_width = min(neg_bin_width, pos_bin_width)

# Ensure bin width is not too small or zero
if bin_width <= 0:
    bin_width = 0.05 * data_range  # Fallback: 5% of data range

# Calculate number of bins
num_bins = int(np.ceil(data_range / bin_width))

# Set minimum number of bins to ensure finer granularity
num_bins = max(40, min(num_bins, 100))  # Increased from 15-50 to 40-100

print(f"Using {num_bins} bins with width {bin_width:.4f}")

# Create bin edges
bin_edges = np.linspace(data_min, data_max, num_bins + 1)

# Calculate histogram heights
neg_heights, _ = np.histogram(neg_values, bins=bin_edges)
pos_heights, _ = np.histogram(pos_values, bins=bin_edges)

# Calculate bin centers for smooth curves
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Normalize heights to make curves comparable (optional)
# Uncomment if you want normalized distributions
# neg_heights = neg_heights / len(neg_values)
# pos_heights = pos_heights / len(pos_values)

# Create smooth curves with increased smoothing
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    # Apply Gaussian smoothing with increased sigma for smoother curves
    # Using higher sigma for smoother curves
    sigma = 2.5  # Increased from 1.2 to 2.5 for smoother curves
    
    # Smooth the heights
    neg_smooth = gaussian_filter1d(neg_heights, sigma=sigma)
    pos_smooth = gaussian_filter1d(pos_heights, sigma=sigma)
    
    # Apply additional smoothing using moving average for even smoother curves
    window_size = 3
    if len(neg_smooth) > window_size * 2:
        neg_smooth_smooth = np.convolve(neg_smooth, np.ones(window_size)/window_size, mode='same')
        pos_smooth_smooth = np.convolve(pos_smooth, np.ones(window_size)/window_size, mode='same')
    else:
        neg_smooth_smooth = neg_smooth
        pos_smooth_smooth = pos_smooth
    
    # Plot only the smooth curves (no bars)
    # Using thicker lines with gradients for better visualization
    plt.plot(bin_centers, neg_smooth_smooth, 'b-', linewidth=3.0, alpha=0.9, 
             label='Negative Distribution', zorder=5)
    plt.plot(bin_centers, pos_smooth_smooth, 'r-', linewidth=3.0, alpha=0.9, 
             label='Positive Distribution', zorder=5)
    
    # Add subtle fill under curves for better visibility (optional)
    plt.fill_between(bin_centers, 0, neg_smooth_smooth, alpha=0.15, color='blue', zorder=1)
    plt.fill_between(bin_centers, 0, pos_smooth_smooth, alpha=0.15, color='red', zorder=1)

# Find peaks on the smoothed curves
neg_max_idx = np.argmax(neg_smooth_smooth)
pos_max_idx = np.argmax(pos_smooth_smooth)

neg_max_bin = bin_centers[neg_max_idx]
pos_max_bin = bin_centers[pos_max_idx]

# Add vertical dashed lines at maximum bins
plt.axvline(x=neg_max_bin, color='blue', linestyle='--', linewidth=2.5, 
            alpha=0.8, label=f'Negative Peak: {neg_max_bin:.2f}', zorder=4)
plt.axvline(x=pos_max_bin, color='red', linestyle='--', linewidth=2.5, 
            alpha=0.8, label=f'Positive Peak: {pos_max_bin:.2f}', zorder=4)

# Mark the peak points on the curves
plt.scatter(neg_max_bin, neg_smooth_smooth[neg_max_idx], color='blue', 
            s=100, zorder=6, edgecolors='darkblue', linewidth=2, alpha=0.9)
plt.scatter(pos_max_bin, pos_smooth_smooth[pos_max_idx], color='red', 
            s=100, zorder=6, edgecolors='darkred', linewidth=2, alpha=0.9)

# Customize plot
plt.xlabel('Prediction Values', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Negative and Positive Samples (Density Curves)', fontsize=14, fontweight='bold')

# Adjust x-axis limits for compact display
# Reduced padding for more compact x-axis
x_padding = data_range * 0.02  # 2% padding instead of fixed padding
plt.xlim(data_min - x_padding, data_max + x_padding)

# Adjust y-axis to start from 0
y_max = max(neg_smooth_smooth.max(), pos_smooth_smooth.max()) * 1.05
plt.ylim(0, y_max)

# Improve legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='blue', linewidth=3.0, label='Negative Distribution'),
    Line2D([0], [0], color='red', linewidth=3.0, label='Positive Distribution'),
    Line2D([0], [0], color='blue', linestyle='--', linewidth=2.5, 
           label=f'Negative Peak: {neg_max_bin:.2f}'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, 
           label=f'Positive Peak: {pos_max_bin:.2f}')
]

plt.legend(handles=legend_elements, fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')

# Adjust layout and display
plt.tight_layout()
plt.savefig("smooth_density_curves.png", dpi=300, bbox_inches='tight')

# Print statistics
print(f"\n=== Distribution Statistics ===")
print(f"Total samples: {len(values)}")
print(f"Negative samples: {len(neg_values)}")
print(f"Positive samples: {len(pos_values)}")
print(f"Number of bins: {num_bins}")
print(f"Bin width: {bin_width:.4f}")
print(f"Negative peak at value: {neg_max_bin:.4f}")
print(f"Positive peak at value: {pos_max_bin:.4f}")
print(f"Value range: [{data_min:.4f}, {data_max:.4f}]")
print(f"Image saved as 'smooth_density_curves.png'")

# Calculate and print separation metrics
peak_separation = abs(pos_max_bin - neg_max_bin)
print(f"\n=== Separation Metrics ===")
print(f"Peak separation: {peak_separation:.4f}")
print(f"Normalized separation: {peak_separation/data_range:.4f}")