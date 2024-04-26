# import numpy as np
# import cv2
#
# def overlay_mask(image, mask, mask_color=(255, 0, 0), alpha=0.5):
#     """
#     Overlay a translucent mask on an image.
#
#     Parameters:
#     image (numpy.ndarray): The background image in RGB format.
#     mask (numpy.ndarray): The mask to overlay. Should be the same size as the image.
#     mask_color (tuple): Color for the mask in (B, G, R) format. Default is red.
#     alpha (float): Translucency level of the mask. Default is 0.5.
#     """
#     # Create a binary mask from the color mask
#     mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     _, binary_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
#
#     # Convert binary mask to a three-channel image
#     mask_rgb = np.zeros_like(image)
#     mask_indices = binary_mask > 0
#     for i in range(3):  # BGR channels
#         mask_rgb[:, :, i] = binary_mask * mask_color[i]
#
#     # Overlay the mask
#     overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
#
#     return overlay
#
# # Example usage
# image = cv2.imread('bk_bm_data/test/images/BM_2001_LBM.png')  # Make sure it's in RGB format
# mask = cv2.imread('bm_finetuned_prompt/iter1_prompt/BM_2001_LBM_mask.png')  # Load mask in color
#
# result = overlay_mask(image, mask)
# cv2.imshow('Overlay', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
#
# image = cv2.imread('bk_bm_data/test/images/BM_2001_LBM.png')
# mask = cv2.imread('bm_finetuned_prompt/iter1_prompt/BM_2001_LBM_mask.png', 0)
# mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# image[mask==255] = (36,255,12)
#
# cv2.imshow('image', image)
# cv2.imshow('mask', mask)
# cv2.waitKey()
#

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
