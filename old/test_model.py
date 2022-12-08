import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)

        # Decoder part
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU()

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU()

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU()

        self.conv13 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        # Encoder part
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x_enc1 = self.maxpool1(x)

        x = self.conv3(x_enc1)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x_enc2 = self.maxpool2(x)

        x = self.conv5(x_enc2)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x_enc3 = self.maxpool3(x)

        # Decoder part
        x = self.upconv1(x_enc3)
        x = torch.cat([x_enc2, x], dim=1)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)

        x = self.upconv2(x)
        x = torch.cat([x_enc1, x], dim=1)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)

        x = self.upconv3(x)
        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)

        x = self.conv13(x)

        return x


