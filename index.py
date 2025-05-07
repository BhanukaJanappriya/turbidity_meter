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

def compute_intensity_metrics(image):
    """
    Compute intensity metrics: average intensity, variance, min, and max.
    """
    avg_intensity = np.mean(image)
    intensity_variance = np.var(image)
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    return avg_intensity, intensity_variance, min_intensity, max_intensity


def segment_regions(image,block_size = 50):
    """
    Divide the image into blocks and compute average brightness for each block.
    
    """
    h, w = image.shape
    brightness_values = []
    for y in range (0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue
            brightness = np.mean(block)
            brightness_values.append(brightness)
    return brightness_values

def fit_exponential_model(intensities, ntu_values):
    """
    Fit an exponential model NTU = a * exp(-b * I_avg) by linearizing the data.
    """
    ln_ntu = np.log(ntu_values)
    coeffs = np.polyfit(intensities,ln_ntu, 1)
    b = -coeffs[0]
    ln_a = coeffs[1]
    a = np.exp(ln_a)
    return a, b

calibration_images =[
    ("0.jpg", 0),
    ("1.jpg", 1),
    ("10.jpg", 10),
    ("100.jpg", 100)
]


calibration_intensities = []
calibration_ntu_values = []

for image_path, ntu_value in calibration_images:
    try:
        preprocessed = preprocess_image(image_path)
        avg_intensity, _, _, _ = compute_intensity_metrics(preprocessed)
        calibration_intensities.append(avg_intensity)
        calibration_ntu_values.append(ntu_value)
    except FileNotFoundError as e:
        print(e)
        
calibration_intensities = np.array(calibration_intensities)
calibration_ntu_values = np.array(calibration_ntu_values)

a_param, b_param = fit_exponential_model(calibration_intensities, calibration_ntu_values)

unknown_image_path = "unknown_sample.jpg"
try:
    preprocessed_unknown = preprocess_image(unknown_image_path)
    avg_intensity, intensity_variance, min_intensity, max_intensity = compute_intensity_metrics(preprocessed_unknown)
    turbidity_index = compute_turbidity_index(min_intensity, max_intensity)
    edge_density = compute_edge_density(preprocessed_unknown)
    region_brightness = segment_regions(preprocessed_unknown)