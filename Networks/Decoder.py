# --------------------------------------------------------------------------------------------------
# ------------------------------------- MODULE :: Base Decoder -------------------------------------
# --------------------------------------------------------------------------------------------------
from . Network import Network

import torch.nn.functional as functional
import torch


# --------------------------------------------------------------------------------------------------
# ---------------------------------------- CLASS :: Decoder ----------------------------------------
# --------------------------------------------------------------------------------------------------
class Decoder(Network):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- Constructor :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.noise = nn.Dropout(0.25)
        self.funct = nn.ReLU()

        self.conv1 = nn.ConvTranspose2d(128, 112, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=112)
        self.conv2 = nn.ConvTranspose2d(112,  96, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=96)
        self.conv3 = nn.ConvTranspose2d(96,   80, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(num_features=80)
        self.conv4 = nn.ConvTranspose2d(80,   64, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.ConvTranspose2d(64,   48, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(num_features=48)
        self.conv6 = nn.ConvTranspose2d(48,   32, kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.BatchNorm2d(num_features=32)
        self.conv7 = nn.ConvTranspose2d(32,   16, kernel_size=3, stride=1, padding=1)
        self.norm7 = nn.BatchNorm2d(num_features=16)
        self.conv8 = nn.ConvTranspose2d(16,    3, kernel_size=3, stride=1, padding=1)


    # ------------------------------------------------------------------------------------------
    # ------------------------------ METHOD :: Forward Activation ------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.noise(self.funct(self.norm1(self.conv1(x))))
        x = functional.interpolate(x, size=(92,   69), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm2(self.conv2(x))))
        x = functional.interpolate(x, size=(152, 114), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm3(self.conv3(x))))
        x = functional.interpolate(x, size=(212, 159), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm4(self.conv4(x))))
        x = functional.interpolate(x, size=(272, 204), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm5(self.conv5(x))))
        x = functional.interpolate(x, size=(332, 249), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm6(self.conv6(x))))
        x = functional.interpolate(x, size=(392, 294), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm7(self.conv7(x))))
        x = functional.interpolate(x, size=(452, 339), mode='bilinear', align_corners=True)

        x = self.conv8(x)
        x = functional.interpolate(x, size=(512, 384), mode='bilinear', align_corners=True)

        return x