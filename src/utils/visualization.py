import numpy as np
import torch
from torchvision.utils import make_grid
import cv2
import matplotlib.pyplot as plt

def plot_boxes(batch: torch.Tensor) -> torch.Tensor:
    """Return a Tensor of cv2 plotted images to be fed into show_batch"""
    images, boxes = batch["image"], batch["boxes"] 
    plotted = []

    for i, image in enumerate(images):
        img = image.permute(1, 2, 0).numpy().astype(np.uint8).copy()
        for box in boxes[i]:
            x, y, w, h = box.numpy().astype(int)
            if x != 0:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plotted.append(img)

    return torch.Tensor(plotted)

def show_batch(images: torch.Tensor, n: int=4):
    """
    Visualize a batch of samples
    images: single batch of cv2 plotted images converted to tensor
    n: number of images to batch and visualize, even number
    """
    
    # Fix clipping my normalizing image values
    images -= images.min()
    images /= images.max()
    # Visualize
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.set_xticks([]); ax.set_yticks([])
    grid = make_grid(images.detach().permute(0, 3, 1, 2)[:n], nrow=2)
    ax.imshow(grid.permute(1, 2, 0))