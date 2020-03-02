import os
import datetime
import logging
import time
import numpy as np
import torch
import cv2
import model.py
import GNN.py


def train(cfg, args):
    arguments = {}
    arguments["iteration"] = 0
    model1 = model.build_model(cfg, arguments, args.local_rank, args.distributed)
    model1.train()




def main():
