import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F



def total_loss(output1, output2):

    output2_transpose = torch.transpose(output2, 0, 1)

    result = torch.matmul(output1, output2_transpose)

    softmax_output = F.softmax(result, dim=1)

    loss = -torch.mean(softmax_output*torch.eye(output1.size[0]))
                      

    return loss


