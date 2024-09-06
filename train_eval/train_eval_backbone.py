import math
import torch
from torch.nn import functional as F
from util.misc import *
import torch.nn as nn
from typing import Iterable
import statistics




#wasn't sure where to put the loss in your arcitucture so i put it here for now 


def custom_loss(constant_instance, test_sample):
    # Compute dot product of global feature
    dot_product = tf.reduce_sum(tf.multiply(constant_instance, test_sample), axis=1)

    # Apply softmax
    softmax_output = tf.nn.softmax(dot_product)

    # Compute loss
    loss = -1 * softmax_output[0]

    return loss

#send in augmented golbal features (2N x 1024)
#send in batchsize N
def total_loss(batch_size, golbal_features):
  #initalize final loss
  final_loss = 0

  #itterate for constant instance
  for i in range(0, batch_size*2 - 2, 2):
    constant_instance = golbal_features[i]

    #itterate for all test samples
    test_sample = [golbal_features[j] for j in range(i + 1, batch_size*2)]

    #add loss to final loss
    final_loss = final_loss + custom_loss(constant_instance, test_sample)

  return final_loss







def train_one_epoch(model: torch.nn.Module, data_loader, optimizer, device, epoch, scaler=None, criterion=None):
    #set in train mode 
    model.train()
    criterion.train()

    loss_list = []

    #integrate through inputs calculate the loss
    for input1, input2 in data_loader:
        input1, input2 = input1.to(device), input2.to(device) # decice is cuda or cpu
        
        outputs = model(input1, input2)
        output_1, output_2 = torch.split(outputs, input.size(0), dim=0)
            
        loss = criterion(output_1, output_2)
        
        #compute the gradient of loss and update paramaters based on gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item()) 
    
    loss_list_avg = statistics.mean(loss_list)
    print(f"Epoch {epoch}, Loss: {loss_list_avg}")
    return loss_list_avg


def evaluate(model: torch.nn.Module, data_loader, device, criterion):
    #set to eval 
    model.eval()
    criterion.eval()
    loss_list = []

    #integrate through inputs to fill list with losses 
    for input1, input2 in data_loader:
        input1, input2 = input1.to(device), input2.to(device)
        
        with torch.no_grad():
            outputs = model(input1, input2)
            output_1, output_2 = torch.split(outputs, input.size(0), dim=0)
            loss = criterion(output_1, output_2)

        loss_list.append(loss.item())
    
    #take and return avrage loss
    loss_list_avg = statistics.mean(loss_list)
    print(f"Loss: {loss_list_avg}")
    return loss_list_avg

        