import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F



def total_loss(output1, output2):
    final_loss = 0

    #take the transpose for the second output matrix
    output2_transpose = torch.transpose(output2, 0, 1)
    
    for i in range(output1.size(0)):
        #take the ith row from the first output matrix 
        row_output = output1[i, :] 
        #compute the product to get the ith row of the product matrix, named result
        #this is done by mulitpying a 1 x dim vector * dim x n Matrix
        result = torch.matmul(row_output, output2_transpose)
        # Apply softmax to the ith row of the product matrix
        softmax_output = nn.softmax(result)
        #take the ith value from ith row, this coresponds to the diagonal 
        loss = -1 * softmax_output[i]

        final_loss = final_loss + loss

    return final_loss


