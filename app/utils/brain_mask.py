import cv2
import numpy as np

def get_brain_mask(image):
    """
    Generate a binary mask for the brain region in the MRI image.
    This is a simple implementation using thresholding. For production,
    consider using a dedicated brain segmentation library like nibabel or a pre-trained model.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's thresholding to create a binary mask
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: Find the largest contour (assuming it's the brain)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    return mask