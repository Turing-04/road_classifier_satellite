from old.old_dataset import dataloader
from old.test_model import UNet
import torch
import torch.nn as nn



model = UNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

for inputs, targets in dataloader():

    # permute to be 1x256x256
    # inputs = inputs.permute(0, 2, 3, 1)

    # fix error  Given groups=1, weight of size [32, 64, 3, 3], expected input[1, 32, 400, 400] to have 64 channels, but got 32 channels instead
    # inputs = inputs.permute(0, 2, 3, 1)


    print(inputs.shape)

    # Compute the model output
    output = model(inputs)


    # Compute the loss
    loss = criterion(output, targets)

    # Backpropagate the gradient
    loss.backward()

    # Update the model parameters
    optimizer.step()

    # Reset the gradient
    optimizer.zero_grad()

    # Print the loss
    print(loss.item())

