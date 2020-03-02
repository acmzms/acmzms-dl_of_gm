import torch
import torch.nn as nn

from GMN.bi_stochastic import BiStochastic
from GMN.voting_layer import Voting
from GMN.displacement_layer import Displacement
from utils.build_graphs import reshape_edge_feature
from utils.feature_align import feature_align
from utils.fgm import construct_m
from NGM.gnn import GNNLayer
from NGM.geo_edge_feature import geo_edge_feature
from GMN.affinity_layer import InnerpAffinity, GaussianAffinity
from model import build_model


class net:
    def __init__(self, cfg, arguments, local_rank, distributed):
        self.sgg_layer = build_model(cfg, arguments, local_rank, distributed)
        self.gnn_layer = GNNLayer(nn.Module)


    def foward(self):


