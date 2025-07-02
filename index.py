import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os


# ------------------------------
# Enhanced Function Definitions
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

def extract_three_regions(image):
    """
    Extract three specific regions from the image for turbidity analysis.
    Returns top, center, and bottom regions.
    """
    h, w, c = image.shape
    
    # Define region boundaries
    region_height = h // 3
    crop_width = min(w // 2, 100)  # Use half width or 100 pixels, whichever is smaller
    center_x = w // 2
    
    # Extract three regions
    regions = {}
    
    # Top region
    top_region = image[0:region_height, 
                     center_x - crop_width//2:center_x + crop_width//2]
    regions['top'] = {
        'image': top_region,
        'position': 'Top Region',
        'bounds': (0, region_height, center_x - crop_width//2, center_x + crop_width//2)
    }
    
    # Center region
    center_region = image[region_height:2*region_height, 
                         center_x - crop_width//2:center_x + crop_width//2]
    regions['center'] = {
        'image': center_region,
        'position': 'Center Region',
        'bounds': (region_height, 2*region_height, center_x - crop_width//2, center_x + crop_width//2)
    }
    
    # Bottom region
    bottom_region = image[2*region_height:h, 
                         center_x - crop_width//2:center_x + crop_width//2]
    regions['bottom'] = {
        'image': bottom_region,
        'position': 'Bottom Region',
        'bounds': (2*region_height, h, center_x - crop_width//2, center_x + crop_width//2)
    }
    
    return regions

def calculate_turbidity_for_region(region_image):
    """
    Calculate turbidity for a specific region using the red channel method.
    """
    if region_image.size == 0:
        return 0, 0, 0
    
    # Calculate mean red channel value (OpenCV uses BGR order)
    m_red = np.mean(region_image[:, :, 2])
    
    # Apply the exponential formula
    turb = -123.03 * np.exp(-m_red / 202.008) - 184.47115 * np.exp(-m_red / 1157.359) + 313.5892
    turbidity_out = round(-10.03 * turb + 1274.35)
    
    # Ensure non-negative values
    turbidity_out = max(0, turbidity_out)
    
    return turbidity_out, m_red, turb

def analyze_three_regions(image):
    """
    Analyze turbidity in three regions and return individual and average results.
    """
    regions = extract_three_regions(image)
    region_results = []
    total_ntu = 0
    
    for region_name, region_data in regions.items():
        ntu, red_mean, turb_raw = calculate_turbidity_for_region(region_data['image'])
        
        result = {
            'name': region_data['position'],
            'ntu': ntu,
            'red_mean': red_mean,
            'turb_raw': turb_raw,
            'bounds': region_data['bounds']
        }
        
        region_results.append(result)
        total_ntu += ntu
    
    average_ntu = total_ntu / len(region_results)
    
    return {
        'regions': region_results,
        'average_ntu': average_ntu,
        'total_regions': len(region_results)
    }

def calculate_relative_turbidity(reference_analysis, sample_analysis):
    """
    Calculate device-independent relative turbidity using division method.
    Reference should be 0 NTU solution.
    """
    # Extract red channel means for all regions
    ref_red_means = [region['red_mean'] for region in reference_analysis['regions']]
    sample_red_means = [region['red_mean'] for region in sample_analysis['regions']]
    
    # Calculate average red channel intensities
    ref_avg_intensity = np.mean(ref_red_means)
    sample_avg_intensity = np.mean(sample_red_means)
    
    # Method 1: Intensity ratio method
    if ref_avg_intensity == 0:
        intensity_ratio = 1
    else:
        intensity_ratio = sample_avg_intensity / ref_avg_intensity
    
    # Method 2: Relative turbidity calculation
    # Since clearer water (0 NTU) should have higher red channel intensity,
    # turbidity is inversely related to intensity
    if ref_avg_intensity == 0:
        relative_ntu = sample_analysis['average_ntu']
    else:
        # Use intensity difference normalized by reference
        intensity_difference = ref_avg_intensity - sample_avg_intensity
        relative_ntu = max(0, (intensity_difference / ref_avg_intensity) * 100)
        
        # Alternative calculation using logarithmic relationship
        if sample_avg_intensity > 0:
            log_ratio = np.log(ref_avg_intensity / sample_avg_intensity)
            relative_ntu_log = max(0, log_ratio * 50)  # Scale factor for practical range
        else:
            relative_ntu_log = 100
        
        # Take average of both methods for more robust result
        relative_ntu = (relative_ntu + relative_ntu_log) / 2
    
    # Method 3: Regional analysis for consistency check
    regional_ratios = []
    for ref_region, sample_region in zip(reference_analysis['regions'], sample_analysis['regions']):
        if ref_region['red_mean'] > 0:
            ratio = sample_region['red_mean'] / ref_region['red_mean']
            regional_ratios.append(ratio)
    
    regional_consistency = np.std(regional_ratios) if regional_ratios else 0
    
    return {
        'relative_ntu': round(relative_ntu, 2),
        'intensity_ratio': round(intensity_ratio, 3),
        'reference_intensity': round(ref_avg_intensity, 2),
        'sample_intensity': round(sample_avg_intensity, 2),
        'regional_consistency': round(regional_consistency, 3),
        'method': 'device_independent_division'
    }

# ------------------------------
# Main Analysis Functions
# ------------------------------

def analyze_single_image(image_path):
    """
    Analyze a single image and return final turbidity value.
    """
    try:
        # Preprocess the image
        preprocessed, original = preprocess_image(image_path)
        
        # Perform three-region analysis
        region_analysis = analyze_three_regions(original)
        
        # Return only the final turbidity value as requested
        return {
            'image_path': image_path,
            'final_turbidity_ntu': round(region_analysis['average_ntu'], 2),
            'region_details': region_analysis['regions'],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'image_path': image_path,
            'error': str(e),
            'final_turbidity_ntu': None
        }

def analyze_relative_turbidity(reference_image_path, sample_image_path):
    """
    Analyze relative turbidity between reference (0 NTU) and sample images.
    """
    try:
        # Analyze reference image (0 NTU solution)
        ref_preprocessed, ref_original = preprocess_image(reference_image_path)
        reference_analysis = analyze_three_regions(ref_original)
        
        # Analyze sample image (unknown solution)
        sample_preprocessed, sample_original = preprocess_image(sample_image_path)
        sample_analysis = analyze_three_regions(sample_original)
        
        # Calculate relative turbidity
        relative_result = calculate_relative_turbidity(reference_analysis, sample_analysis)
        
        return {
            'reference_image': reference_image_path,
            'sample_image': sample_image_path,
            'reference_ntu': round(reference_analysis['average_ntu'], 2),
            'sample_absolute_ntu': round(sample_analysis['average_ntu'], 2),
            'relative_turbidity_ntu': relative_result['relative_ntu'],
            'intensity_ratio': relative_result['intensity_ratio'],
            'reference_intensity': relative_result['reference_intensity'],
            'sample_intensity': relative_result['sample_intensity'],
            'regional_consistency': relative_result['regional_consistency'],
            'analysis_timestamp': datetime.now().isoformat(),
            'method': 'Device-Independent Division Method'
        }
        
    except Exception as e:
        return {
            'reference_image': reference_image_path,
            'sample_image': sample_image_path,
            'error': str(e),
            'relative_turbidity_ntu': None
        }

def visualize_regions(image_path, save_plot=False):
    """
    Visualize the three regions used for analysis.
    """
    try:
        _, original = preprocess_image(image_path)
        regions = extract_three_regions(original)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Draw region boundaries on original
        img_with_regions = original.copy()
        h, w = original.shape[:2]
        region_height = h // 3
        crop_width = min(w // 2, 100)
        center_x = w // 2
        
        # Draw rectangles for regions
        cv2.rectangle(img_with_regions, 
                     (center_x - crop_width//2, 0), 
                     (center_x + crop_width//2, region_height), 
                     (0, 255, 0), 2)
        cv2.rectangle(img_with_regions, 
                     (center_x - crop_width//2, region_height), 
                     (center_x + crop_width//2, 2*region_height), 
                     (255, 0, 0), 2)
        cv2.rectangle(img_with_regions, 
                     (center_x - crop_width//2, 2*region_height), 
                     (center_x + crop_width//2, h), 
                     (0, 0, 255), 2)
        
        axes[0, 1].imshow(cv2.cvtColor(img_with_regions, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Regions Marked')
        axes[0, 1].axis('off')
        
        # Individual regions
        region_list = list(regions.values())
        for i, region_data in enumerate(region_list):
            if i < 2:  # Show first two regions
                axes[1, i].imshow(cv2.cvtColor(region_data['image'], cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(region_data['position'])
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'regions_visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {e}")

# ------------------------------
# User Interface Functions
# ------------------------------

def main_menu():
    """
    Main menu for user interaction.
    """
    print("\n" + "="*50)
    print("      TURBIDITY ANALYSIS SYSTEM")
    print("="*50)
    print("1. Analyze Single Image (Absolute NTU)")
    print("2. Analyze Relative Turbidity (0 NTU vs Unknown)")
    print("3. Visualize Analysis Regions")
    print("4. Exit")
    print("="*50)

def get_single_image_analysis():
    """
    Handle single image analysis.
    """
    print("\n--- Single Image Analysis ---")
    image_path = input("Enter the path to your turbidity image: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return
    
    print("Analyzing image...")
    result = analyze_single_image(image_path)
    
    if result['final_turbidity_ntu'] is not None:
        print(f"\n--- ANALYSIS RESULTS ---")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Final Turbidity: {result['final_turbidity_ntu']} NTU")
        print(f"\nRegional Breakdown:")
        for region in result['region_details']:
            print(f"  {region['name']}: {region['ntu']} NTU")
        print(f"Analysis completed at: {result['analysis_timestamp']}")
    else:
        print(f"Error analyzing image: {result['error']}")

def get_relative_analysis():
    """
    Handle relative turbidity analysis.
    """
    print("\n--- Relative Turbidity Analysis ---")
    print("This method reduces device dependency by comparing with a 0 NTU reference.")
    
    ref_path = input("Enter path to reference image (0 NTU solution): ").strip()
    if not os.path.exists(ref_path):
        print(f"Error: Reference file '{ref_path}' not found.")
        return
    
    sample_path = input("Enter path to sample image (unknown solution): ").strip()
    if not os.path.exists(sample_path):
        print(f"Error: Sample file '{sample_path}' not found.")
        return
    
    print("Analyzing images...")
    result = analyze_relative_turbidity(ref_path, sample_path)
    
    if result['relative_turbidity_ntu'] is not None:
        print(f"\n--- RELATIVE ANALYSIS RESULTS ---")
        print(f"Reference Image: {os.path.basename(ref_path)}")
        print(f"Sample Image: {os.path.basename(sample_path)}")
        print(f"\nAbsolute Measurements:")
        print(f"  Reference: {result['reference_ntu']} NTU")
        print(f"  Sample: {result['sample_absolute_ntu']} NTU")
        print(f"\nRelative Measurement:")
        print(f"  RELATIVE TURBIDITY: {result['relative_turbidity_ntu']} NTU")
        print(f"\nTechnical Details:")
        print(f"  Intensity Ratio: {result['intensity_ratio']}")
        print(f"  Reference Intensity: {result['reference_intensity']}")
        print(f"  Sample Intensity: {result['sample_intensity']}")
        print(f"  Regional Consistency: {result['regional_consistency']}")
        print(f"  Method: {result['method']}")
        print(f"Analysis completed at: {result['analysis_timestamp']}")
    else:
        print(f"Error in relative analysis: {result['error']}")

def get_visualization():
    """
    Handle region visualization.
    """
    print("\n--- Region Visualization ---")
    image_path = input("Enter the path to your image for visualization: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return
    
    save_plot = input("Save visualization plot? (y/n): ").strip().lower() == 'y'
    
    print("Generating visualization...")
    visualize_regions(image_path, save_plot)

def run_turbidity_analyzer():
    """
    Main function to run the turbidity analyzer.
    """
    while True:
        main_menu()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            get_single_image_analysis()
        elif choice == '2':
            get_relative_analysis()
        elif choice == '3':
            get_visualization()
        elif choice == '4':
            print("Thank you for using the Turbidity Analysis System!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
        input("\nPress Enter to continue...")

# ------------------------------
# Example Usage Functions
# ------------------------------

def example_single_analysis(image_path):
    """
    Example function for single image analysis.
    """
    result = analyze_single_image(image_path)
    print("Single Image Analysis Result:")
    print(json.dumps(result, indent=2))
    return result

def example_relative_analysis(ref_path, sample_path):
    """
    Example function for relative analysis.
    """
    result = analyze_relative_turbidity(ref_path, sample_path)
    print("Relative Analysis Result:")
    print(json.dumps(result, indent=2))
    return result

# ------------------------------
# Run the application
# ------------------------------

if __name__ == "__main__":
    print("Starting Turbidity Analysis System...")
    print("Make sure you have the required libraries installed:")
    print("pip install opencv-python numpy matplotlib")
    print("\nFor optimal results:")
    print("- Use clear, well-lit images")
    print("- Ensure the sample container is properly centered")
    print("- Use the same lighting conditions for reference and sample images")
    
    # Uncomment the line below to run the interactive system
    run_turbidity_analyzer()
    
    # For direct function calls, use examples like:
    # result = example_single_analysis("path/to/your/image.jpg")
    # result = example_relative_analysis("path/to/reference.jpg", "path/to/sample.jpg")