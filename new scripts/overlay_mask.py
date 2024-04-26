import cv2
import numpy as np

def overlay_translucent_mask(image, mask, color=(255, 0, 0), alpha=0.3):
    """
    Overlay a translucent colored mask on an image.

    Parameters:
    image (numpy.ndarray): The background image.
    mask (numpy.ndarray): The mask to overlay.
    color (tuple): The color for the mask in (B, G, R) format.
    alpha (float): The transparency factor of the mask.
    """
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = color

    # Blend the colored mask with the image
    overlay_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return overlay_image

# Read the image and mask
image = cv2.imread('bk_bm_data/test/images/BM_2001_LBM.png')
mask = cv2.imread('bm_finetuned_prompt/iter1_prompt/BM_2001_LBM_mask.png', 0)
mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Overlay the mask
overlay_image = overlay_translucent_mask(image, mask)

# Display the result
cv2.imshow('Overlay Image', overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
