# road_classifier_satellite

# Dependencies setup

To get a working development environment, you can:
- go to the **utilitary** folder and run this line in anaconda prompt: 

        conda env create -f environment.yml

- use your own environment and install PyTorch with cuda using:
  
        conda install -c pytorch torchvision cudatoolkit=10.1 pytorch


# Dataset setup


### Training data
Create a folder named **training** with the following structure:

    training
    ├── images
    │   └── default
    │   └── expanded
    └── groundtruth
        └── default
        └── expanded

Copy all training images in the **training/images/validation** folder and all groundtruth images in the **training/groundtruth/validation** folder.

### Test data
Create a folder named **test**.
Take all test images out of their subfolders and copy them in the **test** folder.


# Running the code

To get the full training dataset, run:

    python augmentation.py

To get the full testing dataset, run:

    python split_test.py unet
    python split_test.py resnet
    python split_test.py cnn2
    python split_test.py cnn4
    python split_test.py cnn8
    python split_test.py cnn16

To train a model, run:

    python train.py unet
    python train.py resnet
    python train.py cnn2
    python train.py cnn4
    python train.py cnn8
    python train.py cnn16

This will create a Folder weights in which the **trained** models are stored. 

To test a model, run:

    python test.py unet
    python test.py resnet
    python test.py cnn2
    python test.py cnn4
    python test.py cnn8
    python test.py cnn16

To get the full result images, run:

    python unite_results.py unet
    python unite_results.py resnet
    python unite_results.py cnn2
    python unite_results.py cnn4
    python unite_results.py cnn8
    python unite_results.py cnn16

To generate the submission file, run:

    python mask_to_submission.py unet
    python mask_to_submission.py resnet
    python mask_to_submission.py cnn2
    python mask_to_submission.py cnn4
    python mask_to_submission.py cnn8
    python mask_to_submission.py cnn16
