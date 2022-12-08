import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imageio

class InputMaskDataset(Dataset):
    def __init__(self, root_dir, input_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.input_transform = input_transform
        self.mask_transform = mask_transform
        self.input_filenames = os.listdir(os.path.join(root_dir, "images"))
        self.mask_filenames = os.listdir(os.path.join(root_dir, "groundtruth"))

    def __getitem__(self, index):
        # Load the input image and apply the input transformation, if any
        input_path = os.path.join(self.root_dir, "images", self.input_filenames[index])

        input_image = imageio.imread(input_path)

        # crop image to be 256x256
        # input_image = input_image[:256, :256]

        if self.input_transform:
            input_image = self.input_transform(input_image)

        # Load the mask and apply the mask transformation, if any
        mask_path = os.path.join(self.root_dir, "groundtruth", self.mask_filenames[index])

        mask = imageio.imread(mask_path)
        # mask = mask[:256, :256]

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return input_image, mask

    def __len__(self):
        return len(self.input_filenames)


# Define the transforms for the input images and masks
input_transform = transforms.Compose([
    # Apply any transformations to the input images here

    # Convert the input image to a tensor
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    # Apply any transformations to the masks here

    # Convert the mask to a tensor
    transforms.ToTensor(),
])

# Create the dataset instance
dataset = InputMaskDataset(root_dir="training/", input_transform=input_transform, mask_transform=mask_transform)

# Create the data loader for the input images and masks
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    # Specify any other options for the data loader here, such as the batch size
)

def dataloader():
    return data_loader