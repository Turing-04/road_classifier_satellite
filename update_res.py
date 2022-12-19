# for every image in the "training" folder, copy the image in a training_256 folder and resize it to 256x256
import os
from PIL import Image
import glob

# create the folder
if not os.path.exists("training_256"):
    os.makedirs("training_256")
    os.makedirs("training_256/images")
    os.makedirs("training_256/images/default")
    os.makedirs("training_256/images/expanded")
    os.makedirs("training_256/groundtruth")
    os.makedirs("training_256/groundtruth/default")
    os.makedirs("training_256/groundtruth/expanded")

for file in glob.glob("training/images/default/*.png"):
    print("Processing image: " + file)
    im = Image.open(file)
    imResize = im.resize((256,256), Image.ANTIALIAS)
    imResize.save("training_256/images/" + file.split("/")[-1], 'PNG', quality=90)

for file in glob.glob("training/images/expanded/*.png"):
    print("Processing image: " + file)
    im = Image.open(file)
    imResize = im.resize((256,256), Image.ANTIALIAS)
    imResize.save("training_256/images/" + file.split("/")[-1], 'PNG', quality=90)

for file in glob.glob("training/groundtruth/default/*.png"):
    print("Processing groundtruth: " + file)
    im = Image.open(file)
    imResize = im.resize((256,256), Image.ANTIALIAS)
    imResize.save("training_256/groundtruth/" + file.split("/")[-1], 'PNG', quality=90)

for file in glob.glob("training/groundtruth/expanded/*.png"):
    print("Processing groundtruth: " + file)
    im = Image.open(file)
    imResize = im.resize((256,256), Image.ANTIALIAS)
    imResize.save("training_256/groundtruth/" + file.split("/")[-1], 'PNG', quality=90)


