
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
class Contrastor(Network):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- Constructor :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, name: str) -> None:
        super().__init__(name)

    # ------------------------------------------------------------------------------------------
    # ------------------------------ METHOD :: Forward Activation ------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

