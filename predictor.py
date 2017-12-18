import torch
import models

class Predictor:

    def __init__(self):
        # ckpt = torch.load("model.pth")
        # ckpt = torch.load("model.pth")#, map_location=lambda storage, loc: storage)
        ckpt = torch.load("model.pth", map_location=lambda storage, loc: storage)
        self.model = ckpt['net'].eval()

    def __call__(self, data):
        return self.model(data)
