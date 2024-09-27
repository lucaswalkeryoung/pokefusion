# --------------------------------------------------------------------------------------------------
# ------------------------------------- MODULE :: Base Encoder -------------------------------------
# --------------------------------------------------------------------------------------------------
from . Network import Network

import torch.nn.functional as functional
import torch


# --------------------------------------------------------------------------------------------------
# ---------------------------------------- CLASS :: Encoder ----------------------------------------
# --------------------------------------------------------------------------------------------------
class Encoder(Network):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- Constructor :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.noise = nn.Dropout(0.25)
        self.funct = nn.ReLU()

        self.conv1 = nn.Conv2d(3,    16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(16,   32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(32,   48, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(num_features=48)
        self.conv4 = nn.Conv2d(48,   64, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(64,   80, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(num_features=80)
        self.conv6 = nn.Conv2d(80,   96, kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.BatchNorm2d(num_features=96)
        self.conv7 = nn.Conv2d(96,  112, kernel_size=3, stride=1, padding=1)
        self.norm7 = nn.BatchNorm2d(num_features=112)
        self.conv8 = nn.Conv2d(112, 128, kernel_size=3, stride=1, padding=1)
        self.norm8 = nn.BatchNorm2d(num_features=128)


    # ------------------------------------------------------------------------------------------
    # ------------------------------ METHOD :: Forward Activation ------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.noise(self.funct(self.norm1(self.conv1(x))))
        x = functional.interpolate(x, size=(452, 339), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm2(self.conv2(x))))
        x = functional.interpolate(x, size=(392, 294), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm3(self.conv3(x))))
        x = functional.interpolate(x, size=(332, 249), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm4(self.conv4(x))))
        x = functional.interpolate(x, size=(272, 204), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm5(self.conv5(x))))
        x = functional.interpolate(x, size=(212, 159), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm6(self.conv6(x))))
        x = functional.interpolate(x, size=(152, 114), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm7(self.conv7(x))))
        x = functional.interpolate(x, size=(92,   69), mode='bilinear', align_corners=True)

        x = self.noise(self.funct(self.norm8(self.conv8(x))))
        x = functional.interpolate(x, size=(32,   24), mode='bilinear', align_corners=True)

        return x