import math
import torch
from torch.nn import functional as F
from util.misc import *
import torch.nn as nn
from typing import Iterable
import statistics

from util.backbone import total_loss






def train_one_epoch(model: torch.nn.Module, data_loader, optimizer, device, epoch, scaler=None):
    #set in train mode 
    model.train()
    

    loss_list = []

    #integrate through inputs calculate the loss
    for i, (input1, input2) in enumerate(data_loader):
        input1, input2 = input1.to(device), input2.to(device) # decice is cuda or cpu
        
        input_total = torch.cat([input1, input2], 0)
        outputs_ = model(input_total)
        output_1, output_2 = torch.split(outputs_, input1.size[0], dim=0)
            
        loss = total_loss(output_1, output_2)
        
        #compute the gradient of loss and update paramaters based on gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item()) 
        if i % 10:
            print("current batch loss: ", loss.item(), flush=True)
    
    loss_list_avg = statistics.mean(loss_list)
    print(f"Epoch {epoch}, Loss: {loss_list_avg}")
    return loss_list_avg


def evaluate(model: torch.nn.Module, data_loader, device, epoch, scaler=None):
    #set to eval 
    model.eval()
    loss_list = []

    #integrate through inputs to fill list with losses 
    for input1, input2 in data_loader:
        input1, input2 = input1.to(device), input2.to(device)
        
        input_total = torch.cat([input1, input2], 0)


        with torch.no_grad():
            outputs = model(input_total)
            output_1, output_2 = torch.split(outputs, input1.size[0], dim=0)
            loss = total_loss(output_1, output_2)

        loss_list.append(loss.item())
    
    #take and return avrage loss
    loss_list_avg = statistics.mean(loss_list)
    print(f"Epoch {epoch}, Loss: {loss_list_avg}")
    return loss_list_avg

        