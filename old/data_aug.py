import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio
# from albumentations import HorizontalFLip, VerticalFlip, Rotate

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data():
    train_x = sorted(glob(os.path.join("training", "images", "*.png")))
    train_y = sorted(glob(os.path.join("training", "groundtruth", "*.png")))

    test_x = sorted(glob(os.path.join("training", "images", "*.png"))) # TODO: change to test
    test_y = sorted(glob(os.path.join("training", "groundtruth", "*.png"))) # TODO: change to test

    return (train_x, train_y), (test_x, test_y)


def augment_data(images, masks, save_path, augment=True):
    size = (400, 400)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = x.split("/")[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.imread(y)[0]

        if augment == True:
            pass
        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "images", tmp_image_name)
            mask_path = os.path.join(save_path, "groundtruth", tmp_mask_name)

            print(image_path)
            print(mask_path)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            index += 1

        break


if __name__ == "__main__":
    # Seeding
    np.random.seed(42)

    # Load the data
    (train_x, train_y), (test_x, test_y) = load_data()

    # Create the directories
    create_dir("training/augmented/images")
    create_dir("training/augmented/groundtruth")

    # Augment the data
    augment_data(train_x, train_y, "training/augmented", augment=False)



