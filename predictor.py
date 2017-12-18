import torch
import models

import argparse

class Predictor:

    def __init__(self, filename='model.pth'):
        # ckpt = torch.load("model.pth")
        # ckpt = torch.load("model.pth")#, map_location=lambda storage, loc: storage)
        ckpt = torch.load(filename, map_location=lambda storage, loc: storage)
        self.model = ckpt['net'].eval()

    def __call__(self, data):
        return self.model(data)
