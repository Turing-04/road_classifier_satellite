import os
from PIL import Image, ImageEnhance
from utils import create_dir
import random

images_training = "training/images/training/"
images_validation = "training/images/validation/"
groundtruth_training = "training/groundtruth/training/"
groundtruth_validation = "training/groundtruth/validation/"
create_dir(images_training)
create_dir(groundtruth_training)

def create_variations(name, image, folder, apply_filter):
    for i in range(4):
        angle = 90 * i
        im = image.rotate(angle)
        im.save(folder + "rotated_" + str(angle) + "_" + name)

        enhancer = ImageEnhance.Brightness(image)
        im = enhancer.enhance(random.uniform(0.5, 1.5))
        im.save(folder + "rotated_" + str(angle) + "_brightness_" + name)

        enhancer = ImageEnhance.Contrast(image)
        im = enhancer.enhance(random.uniform(0.5, 1.5))
        im.save(folder + "rotated_" + str(angle) + "_contrast_" + name)



images = os.listdir(images_validation)
for name in images:
    print("Processing image: " + name)

    image = Image.open(images_validation + name) 
    create_variations(name, image, images_training, True)

    image_horiz = image.transpose(Image.FLIP_LEFT_RIGHT)
    create_variations("flipped_" + name, image_horiz, images_training, True)



grountruths = os.listdir(groundtruth_validation)
for name in grountruths:
    print("Processing grountruth: " + name)
    
    image = Image.open(groundtruth_validation + name)
    create_variations(name, image, groundtruth_training, False)

    image_horiz = image.transpose(Image.FLIP_LEFT_RIGHT)
    create_variations("flipped_" + name, image_horiz, groundtruth_training, False)
