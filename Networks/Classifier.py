# --------------------------------------------------------------------------------------------------
# -------------------------- MODULE :: Adversarial Encoding Disentangler ---------------------------
# --------------------------------------------------------------------------------------------------
from . Network import Network

import torch.nn.functional as functional
import torch.nn as nn
import torch


# --------------------------------------------------------------------------------------------------
# ---------------------------------- CLASS :: Encoding Classifier ----------------------------------
# --------------------------------------------------------------------------------------------------
class Classifier(Network):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- Constructor :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.noise = nn.Dropout(0.25)
        self.funct = nn.ReLU()

        self.conv1 = nn.Conv2d(128, 112, kernel_size=4, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=112)

        self.conv2 = nn.Conv2d(112,  96, kernel_size=5, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=96)

        self.conv3 = nn.Conv2d(96,   80, kernel_size=5, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(num_features=80)

        self.conv4 = nn.Conv2d(80,   64, kernel_size=5, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(64,   48, kernel_size=5, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(num_features=48)

        self.conv6 = nn.Conv2d(48,   32, kernel_size=5, stride=1, padding=1)
        self.norm6 = nn.BatchNorm2d(num_features=32)

        self.conv7 = nn.Conv2d(32,   16, kernel_size=5, stride=1, padding=1)
        self.norm7 = nn.BatchNorm2d(num_features=16)

        self.conv8 = nn.Conv2d(16,    1, kernel_size=5, stride=1, padding=1)
        self.norm8 = nn.BatchNorm2d(num_features=1)


    # ------------------------------------------------------------------------------------------
    # ------------------------------ METHOD :: Forward Activation ------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.noise(self.funct(self.norm1(self.conv1(x)))) #  4x4
        x = self.noise(self.funct(self.norm2(self.conv2(x)))) #  8x8
        x = self.noise(self.funct(self.norm3(self.conv3(x)))) # 12x12
        x = self.noise(self.funct(self.norm4(self.conv4(x)))) # 16x16
        x = self.noise(self.funct(self.norm5(self.conv5(x)))) # 20x20
        x = self.noise(self.funct(self.norm6(self.conv6(x)))) # 24x24
        x = self.noise(self.funct(self.norm7(self.conv7(x)))) # 28x28
        x = self.noise(self.funct(self.norm8(self.conv8(x)))) # 32x32

        return torch.sigmoid(functional.max_pool2d(x, kernel_size=x.shape[2:]))

