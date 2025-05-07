import numpy as np
import matplotlib.pyplot as plt
import cv2


def preprocess_image(image_path):
    """
    Preprocess the image by converting to grayscale and applying Gaussian blur.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def compute_intensity_metrics(image):
    """
    Apply Canny edge detection and compute edge density

    """
    edges = cv2.Canny(image,100,200)
    edge_pixels = np.sum(edges>0)
    total_pixels = image.shape[0] * image.shape[1]
    edge_density = edge_pixels / total_pixels
    return edge_density

