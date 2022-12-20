import numpy as np
import os, time
from operator import add
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from models import model_unet
from models import model_resnet
from models import model_cnn2
from models import model_cnn4
from models import model_cnn8
from models import model_cnn16

from utils import create_dir, seeding
import sys

# def calculate_metrics(y_true, y_pred):
#     """ Ground truth """
#     y_true = y_true.cpu().numpy()
#     y_true = y_true > 0.5
#     y_true = y_true.astype(np.uint8)
#     y_true = y_true.reshape(-1)

#     """ Prediction """
#     y_pred = y_pred.cpu().numpy()
#     y_pred = y_pred > 0.5
#     y_pred = y_pred.astype(np.uint8)
#     y_pred = y_pred.reshape(-1)

#     score_jaccard = jaccard_score(y_true, y_pred)
#     score_f1 = f1_score(y_true, y_pred)
#     score_recall = recall_score(y_true, y_pred)
#     score_precision = precision_score(y_true, y_pred)
#     score_acc = accuracy_score(y_true, y_pred)

#     return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (400, 400, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (400, 400, 3)
    return mask

if __name__ == "__main__":
    """ Read command line arguments """
    model_name = sys.argv[1]

    """ Setup """
    seeding(42)
    create_dir("results")

    """ Load dataset """
    images = sorted(glob("test/*"))

    """ Define hyperparameters """
    size = (400, 400)

    """ Load the model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == "unet":
        model = model_unet.build_unet()
    elif model_name == "resnet":
        model = model_resnet.build_resnet()
    elif model_name == "cnn2":
        model = model_cnn2.build_cnn2()
    elif model_name == "cnn4":
        model = model_cnn4.build_cnn4()
    elif model_name == "cnn8":
        model = model_cnn8.build_cnn8()
    elif model_name == "cnn16":
        model = model_cnn16.build_cnn16()
    else:
        raise Exception("Please provide a model name")
    model = model.to(device)

    """ Load weights """
    checkpoint_path = "weights/checkpoint_" + model_name + ".pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    # time_taken = []

    for i, x in tqdm(enumerate(images), total=len(images)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (400, 400, 3)
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 400, 400)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 400, 400)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        # """ Reading mask """
        # mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (400, 400)
        # ## mask = cv2.resize(mask, size)
        # y = np.expand_dims(mask, axis=0)            ## (1, 400, 400)
        # y = y/255.0
        # y = np.expand_dims(y, axis=0)               ## (1, 1, 400, 400)
        # y = y.astype(np.float32)
        # y = torch.from_numpy(y)
        # y = y.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            # time_taken.append(total_time)


        #     score = calculate_metrics(y, pred_y)
        #     metrics_score = list(map(add, metrics_score, score))
        #     pred_y = pred_y[0].cpu().numpy()        ## (1, 400, 400)
        #     pred_y = np.squeeze(pred_y, axis=0)     ## (400, 400)
        #     pred_y = pred_y > 0.5
        #     pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        # ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = pred_y * 255
        print(f"results/{name}.png")
        imageio.imwrite(f"results/{name}.png", cat_images)

    # jaccard = metrics_score[0]/len(test_x)
    # f1 = metrics_score[1]/len(test_x)
    # recall = metrics_score[2]/len(test_x)
    # precision = metrics_score[3]/len(test_x)
    # acc = metrics_score[4]/len(test_x)
    # print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    # fps = 1/np.mean(time_taken)
    # print("FPS: ", fps)