import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Function Definitions
# ------------------------------

def preprocess_image(image_path):
    """
    Preprocess the image by converting to grayscale and applying Gaussian blur.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    # Store the original image for red channel analysis
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred, original

def compute_intensity_metrics(image):
    """
    Compute intensity metrics: average intensity, variance, min, and max.
    """
    avg_intensity = np.mean(image)
    intensity_variance = np.var(image)
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    return avg_intensity, intensity_variance, min_intensity, max_intensity

def compute_turbidity_index(min_intensity, max_intensity):
    """
    Calculate the turbidity index based on min and max intensities.
    """
    if (max_intensity + min_intensity) == 0:
        return 0
    return (max_intensity - min_intensity) / (max_intensity + min_intensity)

def compute_edge_density(image):
    """
    Apply Canny edge detection and compute edge density.
    """
    edges = cv2.Canny(image, 100, 200)
    edge_pixels = np.sum(edges > 0)
    total_pixels = image.shape[0] * image.shape[1]
    edge_density = edge_pixels / total_pixels
    return edge_density

def segment_regions(image, block_size=50):
    """
    Divide the image into blocks and compute average brightness for each block.
    """
    h, w = image.shape
    brightness_values = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue
            brightness = np.mean(block)
            brightness_values.append(brightness)
    return brightness_values

def calculate_turbidity_from_red_channel(image):
    """
    Calculate turbidity using the red channel and exponential formula inspired by the reference code.
    """
    # Get image dimensions
    h, w, c = image.shape
    
    # Extract the center crop (100x100 pixels from center)
    center_x, center_y = h // 2, w // 2
    crop_size = 50
    cropped_img = image[center_x-crop_size:center_x+crop_size, 
                        center_y-crop_size:center_y+crop_size]
    
    # Calculate mean red channel value (OpenCV uses BGR order)
    m_red = np.mean(cropped_img[:, :, 2])
    
    # Apply the exponential formula
    turb = -123.03 * np.exp(-m_red / 202.008) - 184.47115 * np.exp(-m_red / 1157.359) + 313.5892
    turbidity_out = round(-10.03 * turb + 1274.35)
    
    return turbidity_out, m_red, turb

def fit_exponential_model(intensities, ntu_values):
    """
    Modified exponential model fitting function that uses a two-term exponential plus constant model.
    This implements a model similar to: a1*exp(-x/b1) + a2*exp(-x/b2) + c
    
    Args:
        intensities: List of intensity values from calibration images
        ntu_values: Corresponding NTU values for the calibration images
        
    Returns:
        Dictionary of model parameters that can be used to predict NTU values
    """
    intensities = np.array(intensities)
    ntu_values = np.array(ntu_values)

    # Filter out invalid values
    valid_mask = ntu_values > 0
    valid_intensities = intensities[valid_mask]
    valid_ntu = ntu_values[valid_mask]

    if len(valid_ntu) < 2:
        raise ValueError("Insufficient valid calibration data for curve fitting.")
    
    # For simplicity, we'll return parameters for both models:
    # 1. Simple exponential: NTU = a * exp(-b * intensity)
    # 2. Complex model: NTU = a1*exp(-x/b1) + a2*exp(-x/b2) + c
    
    # Simple exponential model (linearized fitting)
    ln_ntu = np.log(valid_ntu)
    coeffs = np.polyfit(valid_intensities, ln_ntu, 1)
    b = -coeffs[0]
    ln_a = coeffs[1]
    a = np.exp(ln_a)
    
    # For the complex model, we'll use predefined coefficients
    # These would normally be determined through advanced curve fitting
    # with sufficient calibration data
    complex_model_params = {
        'a1': -123.03,
        'b1': 202.008,
        'a2': -184.47115,
        'b2': 1157.359,
        'c': 313.5892,
        'scale': -10.03,
        'offset': 1274.35
    }
    
    # Return both models
    models = {
        'simple': {'a': a, 'b': b},
        'complex': complex_model_params
    }
    
    return models

# ------------------------------
# Main Processing
# ------------------------------

def process_image(image_path):
    """
    Process a single image and calculate turbidity metrics.
    """
    try:
        # Preprocess the image
        preprocessed, original = preprocess_image(image_path)
        
        # Calculate metrics using grayscale image
        avg_intensity, intensity_variance, min_intensity, max_intensity = compute_intensity_metrics(preprocessed)
        turbidity_index = compute_turbidity_index(min_intensity, max_intensity)
        edge_density = compute_edge_density(preprocessed)
        region_brightness = segment_regions(preprocessed)
        
        # Calculate turbidity using the red channel method
        ntu_value, m_red, turb_value = calculate_turbidity_from_red_channel(original)
        
        # Display the results
        print("\n=== Turbidity Estimation Results ===")
        print(f"Image: {image_path}")
        print(f"Average Intensity (grayscale): {avg_intensity:.2f}")
        print(f"Intensity Variance: {intensity_variance:.2f}")
        print(f"Turbidity Index: {turbidity_index:.4f}")
        print(f"Edge Density: {edge_density:.4f}")
        print(f"Average Red Channel Value: {m_red:.2f}")
        print(f"Turbidity Value (before scaling): {turb_value:.2f}")
        print(f"Estimated NTU: {ntu_value:.2f} NTU")
        
        # Create figures for visualization
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        # Center crop
        h, w, _ = original.shape
        center_x, center_y = h // 2, w // 2
        crop_size = 50
        center_crop = original[center_x-crop_size:center_x+crop_size, center_y-crop_size:center_y+crop_size]
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB))
        plt.title("Center Crop (Used for Red Channel Analysis)")
        plt.axis('off')
        
        # Region brightness histogram
        plt.subplot(2, 2, 3)
        plt.hist(region_brightness, bins=20, color='skyblue', edgecolor='black')
        plt.title("Region Brightness Distribution")
        plt.xlabel("Brightness")
        plt.ylabel("Frequency")
        plt.grid(True)
        
        # Red channel histogram
        plt.subplot(2, 2, 4)
        red_channel = original[:, :, 2]  # OpenCV uses BGR order
        plt.hist(red_channel.flatten(), bins=50, color='red', alpha=0.7)
        plt.axvline(x=m_red, color='black', linestyle='--', label=f'Mean Red: {m_red:.2f}')
        plt.title("Red Channel Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'avg_intensity': avg_intensity,
            'intensity_variance': intensity_variance,
            'turbidity_index': turbidity_index,
            'edge_density': edge_density,
            'm_red': m_red,
            'turb_value': turb_value,
            'ntu_value': ntu_value
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# ------------------------------
# Example Usage
# ------------------------------

# Example calibration data
calibration_images = [
    ("calibration/0.jpg", 0),
    ("calibration/1.jpg", 1),
    ("calibration/10.jpg", 10),
    ("calibration/100.jpg", 100)
]

def calibrate_system(calibration_data):
    """
    Calibrate the system using images with known NTU values.
    """
    print("Calibrating system...")
    calibration_intensities = []
    calibration_ntu_values = []
    
    for image_path, ntu_value in calibration_data:
        try:
            preprocessed, original = preprocess_image(image_path)
            avg_intensity, _, _, _ = compute_intensity_metrics(preprocessed)
            
            # Also get red channel mean for comparison
            h, w, _ = original.shape
            center_x, center_y = h // 2, w // 2
            crop_size = 50
            cropped_img = original[center_x-crop_size:center_x+crop_size, center_y-crop_size:center_y+crop_size]
            m_red = np.mean(cropped_img[:, :, 2])
            
            calibration_intensities.append(avg_intensity)
            calibration_ntu_values.append(ntu_value)
            
            print(f"Calibration image: {image_path}, NTU: {ntu_value}, Avg Intensity: {avg_intensity:.2f}, Red Mean: {m_red:.2f}")
            
        except FileNotFoundError as e:
            print(e)
    
    # Fit exponential model
    if len(calibration_intensities) >= 2:
        model_params = fit_exponential_model(calibration_intensities, calibration_ntu_values)
        print("Calibration complete.")
        return model_params
    else:
        print("Not enough valid calibration images.")
        return None

def main():
    # If you have calibration images, uncomment this
    # models = calibrate_system(calibration_images)
    
    # Process an unknown sample
    unknown_image_path = "calibration/100.jpg"
    results = process_image(unknown_image_path)
    
    if results:
        print("\nProcessing complete.")

if __name__ == "__main__":
    main()