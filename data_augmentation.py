import os
from PIL import Image, ImageEnhance
from skimage.util import random_noise
import random
import numpy as np

# Set the location of the training images
train_dir = 'training/images/default'

# Set the location where the augmented images will be saved
save_dir = 'training/images/expanded'

# Loop through all images in the training folder
for filename in os.listdir(train_dir):

    # Open the image
    im = Image.open(os.path.join(train_dir, filename))
    
    # save the base images
    im.save(os.path.join(save_dir, filename))

    # adjust the brightness of the image
    im = im.point(lambda x: int(x * 1.5))
    im.save(os.path.join(save_dir, 'brightness' + filename))

    # Rotate the image and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.rotate(45)
    im.save(os.path.join(save_dir, 'rotated_45_' + filename))
    
    im = im.rotate(135)
    im.save(os.path.join(save_dir, 'rotated_180_' + filename))

    # Flip the image and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im.save(os.path.join(save_dir, 'flipped_' + filename))


### disabled for performance reasons
        # Add random gaussian noise to the image and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im_array = np.array(im)
    im_array = random_noise(im_array, mode='gaussian')
    # Convert the floating-point array to an integer array
    im_array = (im_array * 255).astype(np.uint8)
    im = Image.fromarray(im_array)
    im.save(os.path.join(save_dir, 'noisy_' + filename))
        
    

### disabled for performance reasons
'''
    # Crop the image and save it again
    # First, select a random region of the image
    width, height = im.size
    left = random.randint(0, width - 128)
    top = random.randint(0, height - 128)
    right = left + 128
    bottom = top + 128
    cropped_im = im.crop((left, top, right, bottom))

    # Save the cropped image to the expanded folder
    cropped_im.save(os.path.join(save_dir, 'cropped_' + filename))

    # Shift the colors of the image and save it again
    # First, create an ImageEnhance object for each color channel
    red_enhancer = ImageEnhance.Color(im)
    green_enhancer = ImageEnhance.Color(im)
    blue_enhancer = ImageEnhance.Color(im)
    
# Save the color-shifted image to the expanded folder
im.save(os.path.join(save_dir, 'color_shifted_' + filename))
'''

# Set the location of the groundtruth images
train_dir = 'training/groundtruth/default'

# Set the location where the augmented images will be saved
save_dir = 'training/groundtruth/expanded'

## Now do the same for there groundtruth equivalent

# Loop through all images in the training folder
for filename in os.listdir(train_dir):

    # Open the image
    im = Image.open(os.path.join(train_dir, filename))
    
    # save the base images
    im.save(os.path.join(save_dir, filename))

    # adjust the brightness of the image
    im = im.point(lambda x: int(x * 1.5))
    im.save(os.path.join(save_dir, 'brightness' + filename))

    # Rotate the image and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.rotate(45)
    im.save(os.path.join(save_dir, 'rotated_45_' + filename))
    
    im = im.rotate(135)
    im.save(os.path.join(save_dir, 'rotated_180_' + filename))

    # Flip the image and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im.save(os.path.join(save_dir, 'flipped_' + filename))


### disable for performance reasons ?
    # We do NOT add noise to the corresponding mask !
    im = Image.open(os.path.join(train_dir, filename))
    im.save(os.path.join(save_dir, 'noisy_' + filename))