# --------------------------------------------------------------------------------------------------
# --------------------------- MODULE :: Detail Enhancer and Text Remover ---------------------------
# --------------------------------------------------------------------------------------------------
from . Network import Network

import torch.nn.functional as functional
import torch.nn as nn
import torch


# --------------------------------------------------------------------------------------------------
# -------------------------------- CLASS :: Enhancer & Text Remover --------------------------------
# --------------------------------------------------------------------------------------------------
class Sharpener(Network):

    # ----------------------------------------------------------------------------------------------
    # -------------------------------- METHOD :: Compute Loss Value --------------------------------
    # ----------------------------------------------------------------------------------------------
    def loss(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(reconstruction, target, reduction='sum')


    # ------------------------------------------------------------------------------------------
    # ------------------------------- Constructor :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.noise = nn.Dropout(0.00)
        self.funct = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = nn.Conv2d(3,   128, kernel_size=4, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=128)

        # self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=1)
        # self.norm3 = nn.BatchNorm2d(num_features=32)
        #
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=1)
        # self.norm4 = nn.BatchNorm2d(num_features=32)
        #
        # self.conv5 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=1)
        # self.norm5 = nn.BatchNorm2d(num_features=32)
        #
        # self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=1)
        # self.norm6 = nn.BatchNorm2d(num_features=32)
        #
        self.conv7 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1, padding=1)
        self.norm7 = nn.BatchNorm2d(num_features=128)

        self.conv8 = nn.ConvTranspose2d(128,   3, kernel_size=4, stride=1, padding=1)


    # ------------------------------------------------------------------------------------------
    # ------------------------------ METHOD :: Forward Activation ------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.noise(self.funct(self.norm1(self.conv1(x)))) #  4x4
        x = self.noise(self.funct(self.norm2(self.conv2(x)))) #  8x8
        # x = self.noise(self.funct(self.norm3(self.conv3(x)))) # 12x12
        # x = self.noise(self.funct(self.norm4(self.conv4(x)))) # 16x16
        # x = self.noise(self.funct(self.norm5(self.conv5(x)))) # 16x16
        # x = self.noise(self.funct(self.norm6(self.conv6(x)))) # 12x12
        x = self.noise(self.funct(self.norm7(self.conv7(x)))) #  8x8

        return torch.sigmoid(self.conv8(x)) # 4x4

