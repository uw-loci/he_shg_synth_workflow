

import os
import sys
import cv2
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'SR-master', 'SR-master', 'CSR-Planet'))
sys.path.append(project_root)
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import shutil
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import rasterio as rio
from rasterio.plot import show as ras_show
from focal_loss import SparseCategoricalFocalLoss
from itertools import product
from tqdm import tqdm
from unet_model import UNet 
from metrics.general import calculate_metrics, generate_CM
from utils.read_tif import LoadData
from utils.show_predictions import VisualizeResults
from PIL import Image

# from augmentation.data_augmentation import augment_data


os.environ['TORCH_HOME'] = '/home_cerberus/speed/henrique.colonese/Segmentation_Teste/'
os.environ['MPLCONFIGDIR'] = '/home_cerberus/speed/henrique.colonese/temp/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class CustomDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = np.transpose(image, (2, 0, 1))  # Change from (H, W, C) to (C, H, W)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask.squeeze(-1), dtype=torch.long)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Apply softmax to inputs to get probabilities
        inputs = F.softmax(inputs, dim=1)
        # One-hot encode the targets
        
        targets = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        # Flatten the tensors for Dice calculation
        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        return 1 - dice

class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets.squeeze(1))
        return ce + dice, ce, dice


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=5):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

def LoadData(path1, path2):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively

    """
    # Read the images folder like a list
    image_dataset = os.listdir(path1)
    mask_dataset = os.listdir(path2)

    # Make a list for images and masks filenames
    orig_img = []
    mask_img = []
    for file in image_dataset:
        if file.endswith('.png'):
            orig_img.append(file)
    for file in mask_dataset:
        if file.endswith('.png'):
            mask_img.append(file)

    # Sort the lists to get both of them in same order (the dataset has exactly the same name for images and corresponding masks)
    orig_img.sort()
    mask_img.sort()

    return orig_img, mask_img

def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):
    """
    Processes the images and mask present in the shared list and path
    Returns a NumPy dataset with images as 3-D arrays of desired size
    Please note the masks in this dataset have only one channel
    """
    # Pull the relevant dimensions for image and mask
    m = len(img)                     # number of images
    i_h, i_w, i_c = target_shape_img   # pull height, width, and channels of image
    m_h, m_w, m_c = target_shape_mask  # pull height, width, and channels of mask

    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m, i_h, i_w, i_c), dtype=np.float32)
    y = np.zeros((m, m_h, m_w, m_c), dtype=np.int32)

    # Resize images and masks
    index = 0
    contador = 0
    for file in img:
        # convert image into an array of desired shape (3 channels)
        path = os.path.join(path1, file)
        with rio.open(path) as src:
            rast_img = src.read()
            # Alterar channels para o final
            rast_img = np.moveaxis(rast_img, 0, 2)
            X[index] = rast_img[:, :, 0:3]
            for i in range(3):
                # print(max(map(max, X[index,i])))
                # print()
                X[index, :, :, i] /= max(map(max, X[index, :, :, i]))
            index += 1
        contador += 1
        if (contador) % 500 == 0:
            print(f"preprocessando imagem: {contador}")

    index = 0
    contador = 0
    for file in mask:
        path = os.path.join(path2, file)
        with rio.open(path) as src:
            rast_mask = src.read()
            rast_mask = np.moveaxis(rast_mask, 0, 2)
            y[index] = rast_mask
            index += 1
        contador += 1
        if (contador) % 500 == 0:
            print(f"preprocessando mask: {contador}")
    return X, y




def check_mask_labels(masks_folder, num_classes):
    for filename in os.listdir(masks_folder):
        mask_path = os.path.join(masks_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_labels = np.unique(mask)
            # print(f"File: {filename}, Unique labels: {unique_labels}")
            if any(label < 0 or label >= num_classes for label in unique_labels):
                print(f"Error: File {filename} contains labels outside the range [0, {num_classes-1}]")


def remap_mask_classes(masks_folder, old_classes, new_classes):
    for filename in os.listdir(masks_folder):
        mask_path = os.path.join(masks_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            # Remap all classes
            for old_class, new_class in zip(old_classes, new_classes):
                mask[mask == old_class] = new_class
            
            # Save the corrected mask
            cv2.imwrite(mask_path, mask)
            # print(f"Processed: {filename}")

def copy_folder(source_folder, destination_folder):
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # os.makedirs(destination_folder, exist_ok=True)
    
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        destination_item = os.path.join(destination_folder, item)
        
        shutil.copy2(source_item, destination_item)
        
    print(f"Copied successfully.")

def limpeza_tile_0(masks_path_5,imgs_path_5,contador_tile_0):
    for filename in os.listdir(masks_path_5):
        # print(f"dir filepath: {os.listdir(masks_path_5)}")
        mask_path = os.path.join(masks_path_5, filename)
        # print(f"mask filepath: {mask_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if (mask == 0).any():
                img_path = os.path.join(imgs_path_5, filename)
                if os.path.exists(img_path):
                  os.remove(img_path)
                os.remove(mask_path)
                contador_tile_0 +=1


def zero_div(x, y):
    if x == 0 and y == 0:
        return 0
    else:
        return x/y

def pixel_accuracy(output, mask):
    with torch.no_grad():

        predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct_predictions = (predictions == mask)
        accuracy = torch.sum(correct_predictions).item() / correct_predictions.numel()

    return accuracy

def m_iou(pred_mask, mask, smooth=1e-10, num_classes=5):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(num_classes):
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().item()
                union = torch.logical_or(true_class, true_label).sum().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)

        # Compute the mean IoU, ignoring NaN values
        return np.nanmean(iou_per_class)


def calculate_balance(folder_path):
    class_counts = {}
    total_pixels = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  
            
            mask_path = os.path.join(folder_path, filename)
            
            
            mask = np.array(Image.open(mask_path))
            total_pixels += mask.size
            
            unique, counts = np.unique(mask, return_counts=True)
            
            
            for cls, count in zip(unique, counts):
                if cls in class_counts:
                    class_counts[cls] += count
                else:
                    class_counts[cls] = count

    class_percentages = {cls: (count / total_pixels) * 100 for cls, count in class_counts.items()}
    classes = sorted(class_percentages.keys())
    percentages = [class_percentages[cls] for cls in classes]
    print("Classes:", classes)
    print("Counts:", percentages)
    
    for cls, percentage in zip(classes, percentages):
        print(f"Class {cls}: {percentage:.2f}%")
    
def generate_CM(model_path,val_loader, classes):
    
    CM = np.zeros((classes, classes))
    
    model = AttU_Net(img_ch=3, output_ch=5)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    
    # model = torch.load(model)
    model.to('cuda')
    model.eval()
    
    model_parameters_on_gpu = [param.to('cuda') for param in model.parameters()]
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flattened_predicted_masks = []
    flattened_masks = []
    
    with torch.no_grad():
        for image, mask in val_loader:
            #image, mask = image.to(device), mask.to(device)
            image, mask = image.to('cuda'), mask.to('cuda')
            predicted_mask = model(image)
            predicted_mask = torch.argmax(predicted_mask, dim=1)

            flattened_predicted_masks.append(predicted_mask.view(predicted_mask.size(0), -1).cpu().numpy())
            flattened_masks.append(mask.view(mask.size(0), -1).cpu().numpy())

    flattened_predicted_masks = np.concatenate(flattened_predicted_masks).flatten()
    flattened_masks = np.concatenate(flattened_masks).flatten()
        
    cm = confusion_matrix(flattened_masks, flattened_predicted_masks)

    uniques = np.unique(flattened_predicted_masks)

    aux_i = 0
    aux_j = 0
    for i in uniques:
        aux_j = 0
        for j in uniques:
            CM[i, j] += cm[aux_i, aux_j]
            aux_j += 1
        aux_i += 1

    return CM

def calculate_metrics(val_loader, classes):
    
    CM = generate_CM('best_model.pth', val_loader, classes)
    
    # Micro f1-score
    micro_f1 = (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3] + CM[4][4]) / \
            (sum(CM[0])+sum(CM[1])+sum(CM[2])+sum(CM[3])+sum(CM[4]))
    
    # Recall and Precision by class, Macro f1-score
    precision = [0, 0, 0, 0, 0]
    recall = [0, 0, 0, 0, 0]
    f1 = [0, 0, 0, 0, 0]

    c = 0
    while c < 5:
        precision[c] = zero_div(CM[c][c], (CM[0][c] + CM[1][c] + CM[2][c] + CM[3][c] + CM[4][c]))  # TP / (TP+FP)
        recall[c] = zero_div(CM[c][c], (CM[c][0] + CM[c][1] + CM[c][2] + CM[c][3] + CM[c][4]))  # TP / (TP+FN)
        f1[c] = zero_div((2*precision[c]*recall[c]), (precision[c]+recall[c]))
        c += 1

    # Macro f1-score
    macro_f1 = sum(f1)/len(f1)

    print("Micro f1-score: ", round(micro_f1, 3))
    print("Precisão por classe: ", [round(elem, 3) for elem in precision])
    print("Recall por classe: ", [round(elem, 3) for elem in recall])
    print("f1-score por classe: ", [round(elem, 3) for elem in f1])
    print("Macro f1-score: ", round(macro_f1, 3))
    
    return CM

# arrumar ainda
def plot_metrics(model_name):
    
    history = torch.load(os.path.join(cs.MODELS_PATH, model_name, 'last_checkpoint.pth.tar'),
                         map_location=torch.device('cuda'))    
        
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['test_losses'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    
    fig1.savefig(os.path.join(cs.MODELS_PATH, model_name, 'loss_history.png'))

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['test_iou'], label='Test IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Test IoU Scores')
    plt.legend()

    fig2.savefig(os.path.join(cs.MODELS_PATH, model_name, 'iou_history.png'))

    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracies')
    plt.legend()

    fig3.savefig(os.path.join(cs.MODELS_PATH, model_name, 'acc_history.png'))

    fig4 = plt.figure(figsize=(10, 5))
    plt.plot(history['lrs'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')

    fig4.savefig(os.path.join(cs.MODELS_PATH, model_name, 'lr_history.png'))
    plt.close()

print("Processo iniciado, realizando backup dos folders")

# obtendo diretorios
dataset_dir = 'CrowdsourcingDataset-Amgadetal2019'
out_masks_path = dataset_dir + '/dataset/masks_splited/'
out_imgs_path = dataset_dir + '/dataset/images_splited/'


# criando folder backup 5 classes
imgs_path_5 = dataset_dir + '/dataset/images_splited_5/'
masks_path_5 = dataset_dir + '/dataset/masks_splited_5/'

# criando variaveis de controle
contador_image = 0
contador_mask = 0
contador_tile_0 = 0

#  criando lista de remap
old_classes = [1, 2, 3, 4, 5]
new_classes = [0, 1, 2, 3, 4]

lista_5_classes = {
    0: ["tumor"],
    1: ["stroma"],
    2: ["lymphocytic_infiltrate"],
    3: ["necrosis_or_debris"],
    4: ["other"]
}

# Crianco variaveis de config model
num_classes = 5
tam_batch = 32

if not os.path.exists(imgs_path_5):
    os.makedirs(imgs_path_5)
    copy_folder(out_imgs_path, imgs_path_5)
    print("imgs folder copiado")

if not os.path.exists(masks_path_5):
    os.makedirs(masks_path_5)
    copy_folder(out_masks_path, masks_path_5)
    print("masks folder copiado")
    limpeza_tile_0(masks_path_5,imgs_path_5,contador_tile_0)
    print("Limpeza tile 0 concluida")
    remap_mask_classes(masks_path_5, old_classes, new_classes)   
    print("Remap de classes concluido")

for filename in os.listdir(imgs_path_5):
        mask_path = os.path.join(imgs_path_5, filename)
        contador_image += 1

for filename in os.listdir(masks_path_5):
        mask_path = os.path.join(masks_path_5, filename)
        contador_mask += 1

print(f"Quantidade de images: {contador_image}")
print(f"Quantidade de masks: {contador_mask}")
print(f"Quantidade de tile com 0: {contador_tile_0}")





# print("Realizando remap de classes")
# remap_mask_classes(masks_path_5, old_classes, new_classes)
print("Realizando checagem de classes")
check_mask_labels(masks_path_5, num_classes)

print("Checando balanceamento das classes")
calculate_balance(masks_path_5)


# Call the apt function
path1 = imgs_path_5
path2 = masks_path_5
print("Starting...")
print("Load data...")
img, mask = LoadData(path1, path2)

print(f"Shape of images to process: {len(img)}")
print(f"Shape of masks to process: {len(mask)}")

target_shape_img = [256, 256, 3]
target_shape_mask = [256, 256, 1]

print("Process data...")
# Process data using apt helper function
X, y = PreprocessData(img, mask, target_shape_img,
                      target_shape_mask, path1, path2)

print("Preprocessing feito.")

X_aux, X_test, y_aux, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_aux, y_aux, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_valid.shape}, {y_valid.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")



"""Fazer Augmentation e arrumar funções de Loss"""
model = AttU_Net(img_ch=3, output_ch=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


train_dataset = CustomDataset(X_train, y_train)
valid_dataset = CustomDataset(X_valid, y_valid)
test_dataset = CustomDataset(X_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


criterion = ComboLoss()
ce_loss_fn = nn.CrossEntropyLoss()
dice_loss_fn = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

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

results_dir = "results_tests"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print("Creating Result folders")
current_time = datetime.now()
date_str = current_time.strftime("%Y%m%d_%H%M")
output_dir = f"data_output_{date_str}"

test_dir = os.path.join(results_dir, output_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)


print("Creating confusion matrix")
CM = calculate_metrics(valid_loader,5)
df_cm = pd.DataFrame(CM,index=list(lista_5_classes.keys()),columns=list(lista_5_classes.keys()))
    
sn.set(font_scale=1.2)
fig2 = plt.figure(figsize=(16,16))
ax = plt.axes()
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt ='.2f', cmap="flare", robust=True, cbar=True, ax=ax)
ax.set_title(lista_5_classes)
fig2.savefig(os.path.join(test_dir, f'confusion_matrix.png'))
plt.close(fig2)

# Visualization
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

total_test_loss, total_test_ce_loss, total_test_dice_loss = 0.0, 0.0, 0.0
total_test_iou, total_test_accuracy = 0.0, 0.0
num_batches = 0

with torch.no_grad():
    for i, (image, mask) in enumerate(test_loader):
        image, mask = image.to(device), mask.to(device)
        output = model(image)
        
        loss, ce_loss, dice_loss = criterion(output, mask)
        iou_score = m_iou(output, mask)
        accuracy = pixel_accuracy(output, mask)
        
        total_test_loss += loss.item()
        total_test_ce_loss += ce_loss.item()
        total_test_dice_loss += dice_loss.item()
        total_test_iou += iou_score
        total_test_accuracy += accuracy
        num_batches += 1
        
        pred_mask = torch.argmax(output, dim=1)
        print(f'Predicted Mask -- Loss: {loss.item():.3f}, CE Loss: {ce_loss.item():.3f}, Dice Loss: {dice_loss.item():.3f}, IoU: {iou_score:.3f}, Pixel Accuracy: {accuracy:.3f}\n')
        
        if(i<=5):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Original Image
            axs[0].imshow(image[0].cpu().numpy().transpose(1, 2, 0))
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            # True Mask
            axs[1].imshow(mask[0].cpu().numpy().squeeze(), cmap='gray')
            axs[1].set_title('True Mask')
            axs[1].axis('off')

            # Predicted Mask
            axs[2].imshow(pred_mask[0].cpu().numpy().squeeze(), cmap='gray')
            axs[2].set_title(f'Predicted Mask\nLoss: {loss.item():.3f}CE Loss: {ce_loss.item():.3f}, Dice Loss: {dice_loss.item():.3f}, IoU: {iou_score:.3f}, Pixel Accuracy: {accuracy:.3f}')
            axs[2].axis('off')

            # Save the combined image
            plt.savefig(os.path.join(test_dir, f'test_image_{i}.png'))
            plt.close(fig)

test_avg_loss = total_test_loss / num_batches
test_avg_ce_loss = total_test_ce_loss / num_batches
test_avg_dice_loss = total_test_dice_loss / num_batches
test_avg_iou_score = total_test_iou / num_batches
test_avg_accuracy = total_test_accuracy / num_batches

print(f"Modelo treinado com {num_classes} classes, batch de tamanho {tam_batch}\n")
print(f"Validação: Loss: {best_loss:.3f}, CE Loss: {best_loss_ce:.3f}, Dice Loss: {best_loss_dice:.3f}, IoU: {best_iou_score:.3f}, Pixel Accuracy: {best_accuracy:.3f}\n")
print(f'Teste: Loss: {test_avg_loss:.3f}, CE Loss: {test_avg_ce_loss:.3f}, Dice Loss: {test_avg_dice_loss:.3f}, IoU: {test_avg_iou_score:.3f}, Pixel Accuracy: {test_avg_accuracy:.3f}\n')

print("Testing complete.")