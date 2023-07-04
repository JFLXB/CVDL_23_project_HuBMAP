import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import Transforms as tr
from torchvision import transforms



def create_test_image(size):
    # Create a synthetic RGB image with distinct color quadrants
    quadrant_size = size // 2
    red_quadrant = np.full((quadrant_size, quadrant_size, 3), [255, 0, 0], dtype=np.uint8)
    green_quadrant = np.full((quadrant_size, quadrant_size, 3), [0, 255, 0], dtype=np.uint8)
    blue_quadrant = np.full((quadrant_size, quadrant_size, 3), [0, 0, 255], dtype=np.uint8)
    white_quadrant = np.full((quadrant_size, quadrant_size, 3), [255, 255, 255], dtype=np.uint8)

    top_half = np.concatenate([red_quadrant, green_quadrant], axis=1)
    bottom_half = np.concatenate([blue_quadrant, white_quadrant], axis=1)
    
    img = np.concatenate([top_half, bottom_half], axis=0)
    
    return Image.fromarray(img, 'RGB')

# def create_test_image(size):
#     # Create a synthetic RGB image with color gradients
#     r = np.linspace(0, 1, size)
#     g = np.linspace(0, 1, size)[::-1]
#     b = np.ones_like(r) * 0.5  # fixed color for blue channel
    
#     rgb = np.stack([r, g, b], axis=2)  # combine color channels
#     rgb = np.tile(rgb, (size, 1, 1))  # repeat the pattern along y-axis

#     # Make 4 distinct quadrants by flipping the color gradients
#     rgb_top = np.hstack([rgb, rgb[:, ::-1, :]])
#     rgb_bottom = np.hstack([rgb[::-1, :, :], rgb])
#     rgb_img = np.vstack([rgb_top, rgb_bottom]) * 255
    
#     return Image.fromarray(rgb_img.astype('uint8'), 'RGB')

# def create_test_image(size):
#     # Create an image with distinct color quadrants.
#     top = np.hstack((np.ones((size // 2, size // 2, 3)), np.zeros((size // 2, size // 2, 3))))
#     bottom = np.hstack((np.zeros((size // 2, size // 2, 3)), np.ones((size // 2, size // 2, 3))))
#     img = np.vstack((top, bottom)) * 255
#     return Image.fromarray(img.astype('uint8'), "RGB")

def visualize_transformation(transformation, trans_name, size=256):
    # img = create_test_image(size)
    img = Image.open("dog.jpeg")
    mask = img.copy()  # For this simple test, we'll just use the same image as mask.
    
    # Apply transformation
    img_transformed, mask_transformed = transformation(img, mask)

    # Plot original and transformed images side by side
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[1].imshow(img_transformed)
    ax[1].set_title(f'{trans_name} Image')
    ax[2].imshow(mask_transformed)
    ax[2].set_title(f'{trans_name} Mask')
    plt.show()


# resize = tr.Resize(128)
# visualize_transformation(resize, "Resize")

# flip = tr.RandomHorizontalFlip(1)
# visualize_transformation(flip, "RandomHorizontalFlip")

# vflip = tr.RandomVerticalFlip(1) 
# visualize_transformation(vflip, "RandomVerticalFlip")

# crop = tr.RandomCrop(128)
# visualize_transformation(crop, "RandomCrop")

hue_sat = tr.RandomHueSaturationValue()
visualize_transformation(hue_sat, "HueSat")

# gamma = tr.RandomGamma()
# visualize_transformation(gamma, "Gamma")



#TODO: doenst work idk why
# rotate = tr.RandomRotate90(1) 
# visualize_transformation(rotate, "RandomRotate90")
