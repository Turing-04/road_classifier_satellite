import os
from PIL import Image, ImageEnhance
from utils import create_dir
import random



create_dir("training/images/expanded")
create_dir("training/groundtruth/expanded")
images_expanded = "training/images/expanded/"
groundtruth_expanded = "training/groundtruth/expanded/"

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



images = os.listdir("training/images/default/")
for name in images:
    print("Processing image: " + name)

    image = Image.open("training/images/default/" + name) 
    create_variations(name, image, images_expanded, True)

    image_horiz = image.transpose(Image.FLIP_LEFT_RIGHT)
    create_variations("flipped_" + name, image_horiz, images_expanded, True)



grountruths = os.listdir("training/groundtruth/default/")
for name in grountruths:
    print("Processing grountruth: " + name)
    
    image = Image.open("training/groundtruth/default/" + name)
    create_variations(name, image, groundtruth_expanded, False)

    image_horiz = image.transpose(Image.FLIP_LEFT_RIGHT)
    create_variations("flipped_" + name, image_horiz, groundtruth_expanded, False)
