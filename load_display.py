import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets



outdist_results = np.load('storage/outdist.npy')
distance_results = np.load('storage/distance.npy')

# Now plot the phase space diagram
plt.figure(figsize=(5, 4))
if len(outdist_results) > 0:
  sc = plt.scatter(outdist_results[:, 0], outdist_results[:, 1], color='red', s=10, marker="c")
if len(distance_results) > 0:
  sc = plt.scatter(distance_results[:, 0], distance_results[:, 1], c=distance_results[:, 2], cmap='viridis', s=10, marker="c")


print("pre results")
print(outdist_results)
print(distance_results)
plt.colorbar(sc, label='Distance to Target (yards)')  # Add colorbar to represent the distance
plt.xlabel('Launch Angle (degrees)')
plt.ylabel('Roll Angle (degrees)')
plt.title('Phase Space Diagram: Launch Angle vs. Roll Angle')

# Show the plot
plt.show()