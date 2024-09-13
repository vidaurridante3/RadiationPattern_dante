#pushed from vs

from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from util.misc import NestedTensor

from .transformer import Transformer_Encoder, Transformer_Decoder, DropPath
from .positional_embedding import build_position_encoding



#variable to have able to change dim_feedforward and encoder layer

# Point Net Backbone
class PointNetBackbone(nn.Module):
    def __init__(self,
                 trans_dim_feedforward = 2048,
                 encoder_num_layers1 = 4,
                 encoder_num_layers2 = 4
                 ):

        super(PointNetBackbone, self).__init__()


        # transformer encoders 1
        self.encoderlayer1 = nn.TransformerEncoderLayer(
            d_model = 3,
            nhead = 1,
            dim_feedforward=trans_dim_feedforward,
            dropout=0.2
        )

        self.transformerencoder1 = nn.TransformerEncoder(
            self.encoderlayer1,
            num_layers = encoder_num_layers1
            )


         # transformer encoders 2
        self.encoderlayer2 = nn.TransformerEncoderLayer(
            d_model = 64,
            nhead = 1,
            dim_feedforward=trans_dim_feedforward,
            dropout=0.2
        )

        self.transformerencoder2 = nn.TransformerEncoder(
            self.encoderlayer2,
            num_layers = encoder_num_layers2
            )

        self.pos_emb = nn.Embedding(
            num_embeddings=361,
            embedding_dim=64
            )

        # shared MLP 1
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace = True)
        )


        # shared MLP 2

        self.mlp2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace = True)
        )

        # "max pool" to get the global features
        self.final_conv = nn.Conv1d(in_channels=361,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )


    def forward(self, x: Tensor) -> Tensor:

        # get nested vector size
        data = x.tensors

        #mask = x.mask

        # pass through first encoder layer
        output = self.transformerencoder1(data)

        # pass through first shared MLP
        #into batch_size, num_features, num_points
        output = output.permute(0, 2, 1)
        output = self.mlp1(output)
        #batch_size, num_points, num_features
        output = output.permute(0, 2, 1)

        # pass through second encoder layer
        pos_emb = self.pos_emb
        output_with_pos = output + pos_emb
        output = self.transformerencoder2(output_with_pos)

        # pass through second MLP
        #into batch_size, num_features, num_points
        output = output.permute(0, 2, 1)
        output = self.mlp2(output)
        #batch_size, num_points, num_features
        output = output.permute(0, 2, 1)


        #pass through max pool
        output = self.final_conv(output)

        return output

