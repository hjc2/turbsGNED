import numpy as np
import matplotlib.pyplot as plt

# Load saved data
distance_results = np.load('distance.npy')
outdist_results = np.load('outdist.npy')

# Create figure
plt.figure(figsize=(12, 8))

# Plot out-of-bounds points
if len(outdist_results) > 0:
    plt.scatter(outdist_results[:, 0], outdist_results[:, 1], 
               color='red', s=50, marker="s", label='Out of Bounds')

# Plot in-bounds points with distance heatmap
if len(distance_results) > 0:
    scatter = plt.scatter(distance_results[:, 0], distance_results[:, 1],
                         c=distance_results[:, 2], cmap='viridis', 
                         s=50, marker="s")
    plt.colorbar(scatter, label='Distance to Target (yards)')

plt.xlabel('Launch Angle (degrees)')
plt.ylabel('Roll Angle (degrees)')
plt.title('Disc Flight Parameters vs Distance to Target')
plt.legend()
plt.show()

# Print statistics
if len(distance_results) > 0:
    min_dist_idx = np.argmin(distance_results[:, 2])
    print(f"Best throw parameters:")
    print(f"Launch angle: {distance_results[min_dist_idx, 0]:.1f}°")
    print(f"Roll angle: {distance_results[min_dist_idx, 1]:.1f}°")
    print(f"Distance to target: {distance_results[min_dist_idx, 2]:.1f} yards")