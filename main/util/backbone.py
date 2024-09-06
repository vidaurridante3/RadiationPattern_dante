import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    '''
    Contrastive Learning Loss Function. The input is the output latent vectors from the backbone network.
    The two inputs are the pairs of latent vectors from the same point cloud and different point clouds.
    For each row, x1 and x2 are the latent vectors from the same point cloud, and x1 and x2 are the latent vectors from different point clouds.
    The shape of the x1 and x2 is (batch_size, latent_dim).
    The output is the contrastive loss.
    The contrastive loss is calculated as follows:
        For each row in x1, calculate the distance between x1[i, :] and x2, and the distance between x1[i, :] and the rest of the x1.
        Then the softmax of all the distances is calculated. 
        The first distance is the distance between x1[i, :] and x2[i, :].
        The rest of the distances are the distances between x1[i, :] and the rest of the x1 and the rest of the x2.
        The contrastive loss is the negative log of the softmax of the first distance.
    The distance is calculated as the cosine similarity.
    '''
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, x1, x2):
        # Normalize the vectors to unit vectors for cosine similarity
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)
        
        # Calculate cosine similarity
        # Positive similarities: each x1 with its corresponding x2
        pos_sim = torch.sum(x1_norm * x2_norm, dim=1)
        
        # Negative similarities: each x1 with all x2s (including the matching one) and all x1s (excluding itself)
        neg_sim_x1_x2 = torch.matmul(x1_norm, x2_norm.T)  # Each x1 with all x2
        neg_sim_x1_x1 = torch.matmul(x1_norm, x1_norm.T)  # Each x1 with all x1
        
        # Create masks to zero out the diagonals for neg_sim_x1_x1 and neg_sim_x1_x2
        mask = torch.eye(x1.size(0)).bool().to(x1.device)
        neg_sim_x1_x1.masked_fill_(mask, float('-inf'))  # Use '-inf' to effectively exclude these during softmax
        neg_sim_x1_x2.masked_fill_(mask, float('-inf'))
        
        # Combine all similarities
        all_similarities = torch.cat((pos_sim.unsqueeze(1), neg_sim_x1_x2, neg_sim_x1_x1), dim=1)
        
        # Compute softmax over negative samples and the corresponding positive sample
        softmax_scores = F.softmax(all_similarities, dim=1)
        
        # The contrastive loss is the negative log likelihood of the first distance (positive pair)
        loss = -torch.log(softmax_scores[:, 0] + 1e-9)  # Adding a small value to prevent log(0)
        
        # Calculate the accuracy
        # The accuracy is the number of times the positive pair has the highest similarity
        # divided by the total number of pairs
        accuracy = (torch.argmax(all_similarities, dim=1) == 0).float().mean()
        
        return loss.mean(), accuracy
    