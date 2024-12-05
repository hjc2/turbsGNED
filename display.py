import numpy as np
import matplotlib.pyplot as plt

from field_display import plotfield, plot_x_marker, update_throw
def on_click(event):
   if event.inaxes is not None:
      # Get click coordinates
      roll_data, launch_data = event.xdata, event.ydata
      print(event.xdata, event.ydata)
      # Find closest point
      distances = np.sqrt((distance_results[:, 0] - roll_data)**2 + (distance_results[:, 1] - launch_data)**2)
      closest_idx = np.argmin(distances)

      print(closest_idx)
      if distances[closest_idx] < 0.5:  # Adjust threshold as needed
         update_throw(roll_angle=event.ydata, launch_angle=event.xdata)
         print(f"\nClosest throw parameters:")

      

# Load data and create plot
distance_results = np.load('distance.npy')
outdist_results = np.load('outdist.npy')

fig = plt.figure(figsize=(12, 8))

if len(outdist_results) > 0:
   plt.scatter(outdist_results[:, 0], outdist_results[:, 1],
              color='red', s=50, marker="s", label='Out of Bounds')

if len(distance_results) > 0:
   scatter = plt.scatter(distance_results[:, 0], distance_results[:, 1],
                        c=distance_results[:, 2], cmap='viridis',
                        s=50, marker="s")
   plt.colorbar(scatter, label='Distance to Target (yards)')

plt.xlabel('Launch Angle (degrees)')
plt.ylabel('Roll Angle (degrees)')
plt.title('Disc Flight Parameters vs Distance to Target')
plt.legend()

# Connect click event
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

# Print best throw
if len(distance_results) > 0:
   min_dist_idx = np.argmin(distance_results[:, 2])
   print(f"\nBest throw parameters:")
   print(f"Launch angle: {distance_results[min_dist_idx, 0]:.1f}°")
   print(f"Roll angle: {distance_results[min_dist_idx, 1]:.1f}°")
   print(f"Distance to target: {distance_results[min_dist_idx, 2]:.1f} yards")