import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from skimage import io, img_as_ubyte, morphology, img_as_bool, img_as_float, exposure, color
from skimage.util.shape import view_as_windows
from skimage.util import crop
from skimage.transform import resize, rescale
from PIL import Image
import imagej

import numpy as np
import os, glob, sys
from collections import OrderedDict
import shutil
import argparse
import pandas as pd
import csv
import warnings
warnings.simplefilter("ignore", UserWarning)


from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torch.functional as F
import torch

import model_SHG as md

"""Fazer Augmentation e arrumar funções de Loss"""
model = md.GeneratorUNet(3,1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


num_epochs = 250
best_loss = float('inf')
best_loss_ce = float('inf')
best_loss_dice = float('inf')
best_iou_score = float('inf')
best_accuracy = float('inf')

patience = 25
trigger_times = 0
accumulation_steps = 4


for epoch in range(num_epochs):
    model.train()
    train_loss, train_ce_loss, train_dice_loss = 0.0, 0.0, 0.0
    train_iou_score, train_accuracy = 0.0, 0.0
    # contador de iteracao
    i_loop = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        outputs = model(images)
        # outputs = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
        loss, ce_loss, dice_loss = criterion(outputs, masks)
        loss.backward()

        iou_score = m_iou(outputs, masks)
        accuracy = pixel_accuracy(outputs, masks)
        
        if (i_loop + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        i_loop +=1
        train_loss += loss.item() * images.size(0)
        train_ce_loss += ce_loss.item() * images.size(0)
        train_dice_loss += dice_loss.item() * images.size(0)
        train_iou_score += iou_score * images.size(0)
        train_accuracy += accuracy * images.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_ce_loss /= len(train_loader.dataset)
    train_dice_loss /= len(train_loader.dataset)
    train_iou_score /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    
    model.eval()
    valid_loss, valid_ce_loss, valid_dice_loss = 0.0, 0.0, 0.0
    valid_iou_score, valid_accuracy = 0.0, 0.0
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
            
            loss, ce_loss, dice_loss = criterion(outputs, masks)
            iou_score = m_iou(outputs, masks)
            accuracy = pixel_accuracy(outputs, masks)
            
            
            valid_loss += loss.item() * images.size(0)
            valid_ce_loss += ce_loss.item() * images.size(0)
            valid_dice_loss += dice_loss.item() * images.size(0)
            valid_iou_score += iou_score * images.size(0)
            valid_accuracy += accuracy * images.size(0)
    
    valid_loss /= len(valid_loader.dataset)
    valid_ce_loss /= len(valid_loader.dataset)
    valid_dice_loss /= len(valid_loader.dataset)
    valid_iou_score /= len(valid_loader.dataset)
    valid_accuracy /= len(valid_loader.dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.3f} (CE: {train_ce_loss:.3f}, Dice: {train_dice_loss:.3f}, IoU: {train_iou_score:.3f}, Pixel Accuracy: {train_accuracy:.3f}). Valid Loss: {valid_loss:.3f} (CE: {valid_ce_loss:.3f}, Dice: {valid_dice_loss:.3f}, IoU: {valid_iou_score:.3f}, Pixel Accuracy: {valid_accuracy:.3f})")
    
    # Early stopping
    if valid_loss < best_loss:
        best_loss = valid_loss
        best_loss_ce = valid_ce_loss
        best_loss_dice = valid_dice_loss
        best_iou_score = valid_iou_score
        best_accuracy = valid_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break

print("Training complete.")