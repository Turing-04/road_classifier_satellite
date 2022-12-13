import os
from PIL import Image

images = os.listdir("training/images/default/")
for image in images:
    print("Processing image: " + image)

    im = Image.open("training/images/default/" + image)
    
    im = im.rotate(90)
    im.save("training/images/expanded/rotated_90_" + image)
    im = im.rotate(180)
    im.save("training/images/expanded/rotated_180_" + image)
    im = im.rotate(270)
    im.save("training/images/expanded/rotated_270_" + image)
    im_horiz = im.transpose(Image.FLIP_LEFT_RIGHT)
    im_horiz.save("training/images/expanded/horiz_" + image)
    im = im_horiz.rotate(90)
    im.save("training/images/expanded/rotated_90_horiz_" + image)
    im = im.rotate(180)
    im.save("training/images/expanded/rotated_180_horiz_" + image)
    im = im.rotate(270)
    im.save("training/images/expanded/rotated_270_horiz_" + image)
    im_vert = im.transpose(Image.FLIP_TOP_BOTTOM)
    im_vert.save("training/images/expanded/vert_" + image)
    im = im_vert.rotate(90)
    im.save("training/images/expanded/rotated_90_vert_" + image)
    im = im.rotate(180)
    im.save("training/images/expanded/rotated_180_vert_" + image)
    im = im.rotate(270)
    im.save("training/images/expanded/rotated_270_vert_" + image)

grountruths = os.listdir("training/groundtruth/default/")
for grountruth in grountruths:
    print("Processing grountruth: " + grountruth)
    
    im = Image.open("training/groundtruth/default/" + grountruth)
    
    im = im.rotate(90)
    im.save("training/groundtruth/expanded/rotated_90_" + grountruth)
    im = im.rotate(180)
    im.save("training/groundtruth/expanded/rotated_180_" + grountruth)
    im = im.rotate(270)
    im.save("training/groundtruth/expanded/rotated_270_" + grountruth)
    im_horiz = im.transpose(Image.FLIP_LEFT_RIGHT)
    im_horiz.save("training/groundtruth/expanded/horiz_" + grountruth)
    im = im_horiz.rotate(90)
    im.save("training/groundtruth/expanded/rotated_90_horiz_" + grountruth)
    im = im.rotate(180)
    im.save("training/groundtruth/expanded/rotated_180_horiz_" + grountruth)
    im = im.rotate(270)
    im.save("training/groundtruth/expanded/rotated_270_horiz_" + grountruth)
    im_vert = im.transpose(Image.FLIP_TOP_BOTTOM)
    im_vert.save("training/groundtruth/expanded/vert_" + grountruth)
    im = im_vert.rotate(90)
    im.save("training/groundtruth/expanded/rotated_90_vert_" + grountruth)
    im = im.rotate(180)
    im.save("training/groundtruth/expanded/rotated_180_vert_" + grountruth)
    im = im.rotate(270)
    im.save("training/groundtruth/expanded/rotated_270_vert_" + grountruth)





