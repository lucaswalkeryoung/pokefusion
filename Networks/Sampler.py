# --------------------------------------------------------------------------------------------------
# ---------------------------- MODULE :: VAE Reparameterization Sampler ----------------------------
# --------------------------------------------------------------------------------------------------
from . Network import Network

import torch


# --------------------------------------------------------------------------------------------------
# ---------------------------------------- CLASS :: Sampler ----------------------------------------
# --------------------------------------------------------------------------------------------------
class Sampler(Network):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- Constructor :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.mu = nn.Linear(98304, 98304)
        self.lv = nn.Linear(98304, 98304)


    # ------------------------------------------------------------------------------------------
    # ------------------------------ METHOD :: Forward Activation ------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x  = x.view(x.size(0), -1)
        mu = self.mu(x)
        lv = self.lv(x)
        x  = x.view(-1, 128, 32, 24)

        return x, mu, lv