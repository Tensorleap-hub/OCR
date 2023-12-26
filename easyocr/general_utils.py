import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_bounding_boxes(image, bbox_array):
    """
    Draw bounding boxes on an image.

    Parameters:
    - image: The input image.
    - bbox_array: Numpy array of bounding box coordinates of shape (n_bboxes, 4, 2).

    Returns:
    - None (displays the image with bounding boxes).
    """
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Iterate through bounding boxes and draw rectangles
    for bbox in bbox_array:
        # Reshape bounding box coordinates to (x1, y1, width, height)
        x, y = bbox[:, 0], bbox[:, 1]
        x_min, y_min = np.min(x), np.min(y)
        width, height = np.max(x) - x_min, np.max(y) - y_min

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the axes
        ax.add_patch(rect)

    # Display the image with bounding boxes
    plt.show()
