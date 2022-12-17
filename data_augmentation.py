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
    
    # save the base images with no rotation
    im.save(os.path.join(save_dir,"rotated_0_" + filename))

    # adjust the brightness of the image
    im = ImageEnhance.Brightness(im).enhance(random.uniform(0.7,1.3))
    im.save(os.path.join(save_dir, 'brightness_' + filename))
    
    # play with contrast 
    im = Image.open(os.path.join(train_dir, filename))
    contrast_factor = random.uniform(0.5, 1.5)
    im = ImageEnhance.Contrast(im).enhance(contrast_factor)
    im.save(os.path.join(save_dir, 'contrast_' + filename))

    # Rotate the image and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.rotate(270)
    im.save(os.path.join(save_dir, 'rotated_270_' + filename))
    
    im = Image.open(os.path.join(train_dir, filename))
    im = im.rotate(180)
    im.save(os.path.join(save_dir, 'rotated_180_' + filename))
    
    im = Image.open(os.path.join(train_dir, filename))
    im = im.rotate(90)
    im.save(os.path.join(save_dir, 'rotated_90_' + filename))

    # Adjust the huing of the image
    im = Image.open(os.path.join(train_dir, filename))
    im = im.convert('HSV')
    im = ImageEnhance.Color(im).enhance(random.uniform(-0.1,0.1))
    im = im.convert('RGB')
    im.save(os.path.join(save_dir, 'hue_' + filename))

    #adjust the saturation of the image 
    im = Image.open(os.path.join(train_dir, filename))
    im = im.convert('HSV')
    im.putdata(list(map(lambda x: (x[0], x[1] * 1.5, x[2]), im.getdata())))
    im = im.convert('RGB')
    im.save(os.path.join(save_dir, 'saturation_' + filename))

    # Flip the image horizontally and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im.save(os.path.join(save_dir, 'flipped_' + filename))
    

### disable for performance reasons ?
        # Add random gaussian noise to the image and save it again
    # im = Image.open(os.path.join(train_dir, filename))
    # im_array = np.array(im)
    # im_array = random_noise(im_array, mode='gaussian')
    # # Convert the floating-point array to an integer array
    # im_array = (im_array * 255).astype(np.uint8)
    # im = Image.fromarray(im_array)
    # im.save(os.path.join(save_dir, 'noisy_' + filename))
        
    

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
    
    # save the base images with no rotation
    im.save(os.path.join(save_dir, "rotated_0_" + filename))

    # do not adjust the brightness of the mask
    im.save(os.path.join(save_dir, 'brightness_' + filename))

    # Rotate the image and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.rotate(270)
    im.save(os.path.join(save_dir, 'rotated_270_' + filename))
    
    im = Image.open(os.path.join(train_dir, filename))
    im = im.rotate(180)
    im.save(os.path.join(save_dir, 'rotated_180_' + filename))
    
    im = Image.open(os.path.join(train_dir, filename))
    im = im.rotate(90)
    im.save(os.path.join(save_dir, 'rotated_90_' + filename))

    # Flip the image and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im.save(os.path.join(save_dir, 'flipped_' + filename))


### disable for performance reasons ?
    # We do NOT add noise to the corresponding mask !
    # We do not hue the mask
    # We do not change saturation of the mask
    # we do not change the contrast of the mask
    im = Image.open(os.path.join(train_dir, filename))
   # im.save(os.path.join(save_dir, 'noisy_' + filename))
    im.save(os.path.join(save_dir, 'hue_' + filename))
    im.save(os.path.join(save_dir, 'saturation_' + filename))
    im.save(os.path.join(save_dir, 'constrast_' + filename))
