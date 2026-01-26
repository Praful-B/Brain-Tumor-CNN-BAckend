import cv2
import numpy as np

def get_brain_mask(image_bgr):


    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur (reduce noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Otsu thresholding
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Morphological closing (fill holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Find contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found — invalid MRI image")

    # 6. Keep largest contour (brain)
    largest_contour = max(contours, key=cv2.contourArea)

    # 7. Create mask
    brain_mask = np.zeros_like(gray)
    cv2.drawContours(brain_mask, [largest_contour], -1, 255, thickness=-1)

    return brain_mask


print("Success")