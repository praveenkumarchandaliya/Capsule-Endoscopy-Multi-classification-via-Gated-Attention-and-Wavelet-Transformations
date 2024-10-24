import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F
from functools import partial
import torch.optim as optim
import torch.autograd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
import cv2
import pywt
random.seed(30)

class Encoder(nn.Module):
    def __init__(self, number_of_channels, num_classes, pretrained = True):
        super(Encoder, self).__init__()
        resnet34_4_channel = get_arch(18, number_of_channels)
        model = resnet34_4_channel(pretrained)    
        self.backbone = model
        
        self.oga1 = ODConv2d(64, 64, 3)
        self.oga2 = ODConv2d(64, 64, 3)
        self.oga3 = ODConv2d(128, 128, 3)
        self.oga4 = ODConv2d(256, 256, 3)

    def _freeze_model(self):
        for p in self.backbone.parameters():
            p.required_grad = False

    def forward(self, x):
        outs = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        y = self.oga1(x)
        outs.append(y)

        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        y = self.oga2(x)
        outs.append(y)

        x = self.backbone.layer2(x)
        y = self.oga3(x)
        outs.append(y)

        x = self.backbone.layer3(x)
        y = self.oga4(x)
        outs.append(y)

        x = self.backbone.layer4(x)
        outs.append(x)
        return tuple(outs)

def _make_res_layer(block, num_residual_blocks, in_channels, out_channels, stride=1):
    identity_downsample = None
    layers = []

    if stride != 1 or in_channels != out_channels:
        identity_downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels))

    layers.append(block(in_channels, out_channels, identity_downsample, stride))
    in_channels = out_channels
    in_channels = out_channels

    for i in range(num_residual_blocks - 1):
        layers.append(block(in_channels, out_channels))

    return nn.Sequential(*layers)
            
# Decoder class definition
class DeCoder(nn.Module):
    def __init__(self, number_of_channels):
        super(DeCoder, self).__init__()
        self.in_channels = (64, 64, 129, 258, 516)
        self.out_channels = (516, 258, 129, 64, 64, number_of_channels)
        self.res_layers = []
        self.conv1x1 = []
        self.conv2x2 = []
        self._make_layers()

    def _make_layers(self):
        for i in range(len(self.in_channels)-1, -1, -1):
            res_layer = _make_res_layer(
                            block=BasicBlock,
                            num_residual_blocks=2,
                            in_channels=128 if (i <= 2 and i!=0) else self.in_channels[i],
                            out_channels=self.out_channels[-(i+1)],
                            stride=1)
            out_planes = self.in_channels[i] if i < 2 else int(self.in_channels[i]//2)
            conv2x2 = nn.Sequential(
                nn.Conv2d(self.in_channels[i], out_planes, kernel_size=2, bias=False),
                nn.InstanceNorm2d(out_planes),
                nn.ReLU(inplace=True))

            conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=128 if (i <= 2 and i!=0) else self.in_channels[i],
                          out_channels=self.out_channels[-(i+1)],
                          kernel_size=1,
                          bias=False),
                nn.InstanceNorm2d(self.out_channels[-(i+1)]))
            self.res_layers.append(res_layer)
            self.conv2x2.append(conv2x2)
            self.conv1x1.append(conv1x1)
        self.res_layers = nn.ModuleList(self.res_layers)
        self.conv2x2 = nn.ModuleList(self.conv2x2)
        self.conv1x1 = nn.ModuleList(self.conv1x1)

    def forward(self, x):
        assert len(x) == len(self.in_channels)

        out = x[-1]
        outs = []
        outs.append(out)

        for i in range(len(self.in_channels)):
            out = F.interpolate(out, scale_factor=2, mode='nearest')
            out = F.pad(out, [0, 1, 0, 1])
            out = self.conv2x2[i](out)
            if i < 4:
                out = torch.cat([out, x[-(i+2)]], dim=1)
            identity = self.conv1x1[i](out)
            out = self.res_layers[i](out) + identity
            outs.append(out)
        outs[-1] = torch.tanh(outs[-1])
        return outs

# BasicBlock and ResNet classes for building the ResNet model
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

resnet_models = {18: torchvision.models.resnet18,
                 34: torchvision.models.resnet34,
                 50: torchvision.models.resnet18,
                 101: torchvision.models.resnet101,
                 152: torchvision.models.resnet152}

class Resnet_multichannel(nn.Module):
    def __init__(self, pretrained=True, encoder_depth=18, num_in_channels=3, num_classes=10):
        super().__init__()

        if encoder_depth not in [18, 34, 50, 101, 152]:
            raise ValueError(f"Encoder depth {encoder_depth} specified does not match any existing Resnet models")

        model = resnet_models[encoder_depth](pretrained)

        self.conv1 = self.increase_channels(model.conv1, num_in_channels)

        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc()

        return x

    def increase_channels(self, m, num_channels=None, copy_weights=0):
        new_in_channels = num_channels if num_channels is not None else m.in_channels + 1
        bias = False if m.bias is None else True

        # Creating new Conv2d layer
        new_m = nn.Conv2d(in_channels=new_in_channels,
                          out_channels=m.out_channels,
                          kernel_size=m.kernel_size,
                          stride=m.stride,
                          padding=m.padding,
                          bias=bias)

        # Copying the weights from the old to the new layer
        new_m.weight[:, :m.in_channels, :, :].data[:, :m.in_channels, :, :] = m.weight.clone()

        #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
        for i in range(new_in_channels - m.in_channels):
            channel = m.in_channels + i
            new_m.weight[:, channel:channel+1, :, :].data[:, channel:channel+1, :, :] = m.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_m.weight = nn.Parameter(new_m.weight)

        return new_m

def get_arch(encoder_depth, num_in_channels):
    return partial(Resnet_multichannel, encoder_depth=encoder_depth, num_in_channels=num_in_channels)

class Classifier(nn.Module):

    def __init__(self, num_classes = 10, drop_ratio = 0.5, number_of_channels=3):

        super(Classifier, self).__init__()
        resnet34_4_channel = get_arch(18, number_of_channels)

        # use resnet34_4_channels(False) to get a non pretrained model
        model = resnet34_4_channel(True)
        self.resnet18 = model
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet18.fc(x)
        return x

# Omni-dimensional Gated Attention Mechanism
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        
        # Attention channel size is determined by the reduction ratio
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        # Channel attention
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        # Filter attention (optional for depth-wise convolution)
        if in_planes == groups and in_planes == out_planes:  
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        # Spatial attention (if kernel size > 1)
        if kernel_size == 1:  
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        # Kernel attention (optional if only one kernel is used)
        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    # Channel attention computation
    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    # Filter attention computation
    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    # Spatial attention computation
    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    # Kernel attention computation
    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


# ODConv2d Layer with Omni-dimensional Attention
class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        
        # Attention mechanism for omni-dimensional convolution
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)

        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        # Define different forward methods based on kernel size and number
        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    # Standard forward method for dynamic kernels
    def _forward_impl_common(self, x):
        # Apply attention across all dimensions
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)

        # Calculate the aggregate weight based on attention
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)

        # Summing weights for all kernels
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])

        # Convolution with aggregated weights
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)

        # Reshape the output and apply filter attention
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention

        return output

    # Point-wise convolution (1x1 kernel) forward method
    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        
        # Apply channel attention and perform standard convolution
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        
        # Apply filter attention to the output
        output = output * filter_attention
        return output

    # Main forward method to dispatch the appropriate implementation
    def forward(self, x):
        return self._forward_impl(x)

#Apply Stationary Wavelet Transformation    
def apply_swt(image):
    coeffs = pywt.swt2(image.cpu().numpy(), wavelet='haar', level=1, start_level=0, axes=(-2, -1))
    cA, (cH, cV, cD) = coeffs[0]
    swt_image = torch.cat([torch.tensor(cA), torch.tensor(cH), torch.tensor(cV), torch.tensor(cD)], dim=1)
    return swt_image.to(image.device)

#Apply Discrete Wavelet Transformation
def apply_dwt(image):
    
    cA, (cH, cV, cD) = pywt.dwt2(image.cpu().numpy(), 'haar')
    
    dwt_image = torch.cat([torch.tensor(cA), torch.tensor(cH)], dim=1)
    return dwt_image.to(image.device)

#Apply GrayScale
class ToGrayscale:
    def __call__(self, tensor):
        R, G, B = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]
        grayscale = 0.299 * R + 0.587 * G + 0.114 * B
        return grayscale.unsqueeze(1)

# UNetResNet18 definition
class UNetResNet18(nn.Module):
    def __init__(self):
        super(UNetResNet18, self).__init__()
        self.encoder = Encoder(3, 10)
        self.decoder = DeCoder(3)
        self.resnet = Classifier(num_classes = 10, number_of_channels=3)
        

    def forward(self, x):
        enc_out = self.encoder(x) #Encoder Output
        
        enc_outs = list(enc_out)
        to_grayscale = ToGrayscale()
        swt_image = apply_swt(to_grayscale(x)) #applies SWT
        swt_images_resized = F.interpolate(swt_image, size=enc_outs[-1].shape[2:], mode='nearest')
        
        dwt_image = apply_dwt(to_grayscale(x)) #applies DWT
        dwt_images_resized = F.interpolate(dwt_image, size=enc_outs[-2].shape[2:], mode='nearest')
        
        to_grayscale_resized2 = F.interpolate(to_grayscale(x), size=enc_outs[-3].shape[2:], mode='nearest')
        
        enc_outs[-1] = torch.cat([enc_outs[-1], swt_images_resized], dim=1)    # Concatenate SWT components (4 channels) to (512, 8, 8)
        enc_outs[-2] = torch.cat([enc_outs[-2], dwt_images_resized], dim=1)    # Concatenate DWT components (2 channels) to (256, 16, 16)
        enc_outs[-3] = torch.cat([enc_outs[-3], to_grayscale_resized2], dim=1) # Concatenate GrayScale
        
        dec_outs = self.decoder(enc_outs) #Decoder Output
        
        resnet_out = self.resnet(dec_outs[-1]) #Classifier
        return resnet_out

# Function to save a checkpoint
def save_checkpoint(epoch, model, optimizer, filename='CP-PT_CVIP_2024.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

# Function to load a checkpoint
def load_checkpoint(model, optimizer, filename='CP-PT_CVIP_2024.pth'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

# Calculate metrics
def calculate_acc(gt_y, pred_y, result_file): 
    y_score = np.eye(len(np.unique(gt_y)))[pred_y]

    auc = metrics.roc_auc_score(gt_y, y_score, multi_class='ovr')
    acc = metrics.accuracy_score(gt_y, pred_y)
    recall = metrics.recall_score(gt_y, pred_y, average='weighted')
    f1_score = metrics.f1_score(gt_y, pred_y, average='weighted')
    precision = metrics.precision_score(gt_y, pred_y, average='weighted')
    
    # Confusion matrix for all classes
    cm = metrics.confusion_matrix(gt_y, pred_y)

    # Specificity calculation for multiclass
    specificity_list = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i] 
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    
    # Average specificity across all classes
    specificity = np.mean(specificity_list)
    print("Specificity: ", specificity)
    
    #Balanced Accuracy
    balanced_acc = (recall + specificity) / 2
    print("Recall: ", recall)
    print("Balanced Accuracy: ", balanced_acc)

    print('auc : %.4f, acc : %.4f, precision : %.4f, recall : %.4f, f1_score : %.4f, specificity : %.4f, balanced_acc : %.4f' % \
          (auc, acc, precision, recall, f1_score, specificity, balanced_acc))
    
    auc_report = 'auc : %.4f, acc : %.4f, precision : %.4f, recall : %.4f, f1_score : %.4f, specificity : %.4f, balanced_acc : %.4f' % \
                 (auc, acc, precision, recall, f1_score, specificity, balanced_acc)

    print('%s gt vs. pred %s' % ('-' * 36, '-' * 36))
    print(metrics.classification_report(gt_y, pred_y, digits=4))
    print(metrics.confusion_matrix(gt_y, pred_y))
    print('-' * 85)
    
    result_file = result_file.replace(".txt", ".csv")
    report = metrics.classification_report(gt_y, pred_y, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(result_file, mode='a', index=True)
    
    with open(result_file, 'a') as f:
        f.write("-------------------------------------------------------\n")
        f.write(auc_report)
        f.write("\n")
        f.write(np.array2string(metrics.confusion_matrix(gt_y, pred_y), separator=', '))
        f.write("\n-------------------------------------------------------\n")
    
    return acc

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, checkpoint_file='CP-PT_CVIP_2024.pth'):
    model.train()
    start_epoch = load_checkpoint(model, optimizer, checkpoint_file)
    if start_epoch > 0:
        print(f'Resuming training from epoch {start_epoch + 2}')
        start_epoch=start_epoch+1
    metrics = load_metrics_from_csv()

    # Initialize lists to store metrics
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:  # Print every 100 batches
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        # Store training metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Save the checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer, checkpoint_file)
        print(f'Checkpoint saved at epoch {epoch + 1}')

        val_loss, val_accuracy = validate_model(model, valid_loader, criterion)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_accuracy)

        # Append new metrics to the existing metrics
        metrics['train_losses'].extend(train_losses)
        metrics['valid_losses'].extend(valid_losses)
        metrics['train_accuracies'].extend(train_accuracies)
        metrics['valid_accuracies'].extend(valid_accuracies)

        # Save updated metrics
        save_metrics_to_csv(metrics)

        # Plot the metrics
        plot_metrics(metrics)
        
        # Clear the lists for the next epoch
        train_losses.clear()
        valid_losses.clear()
        train_accuracies.clear()
        valid_accuracies.clear()

# Validation function with return of metrics
def validate_model(model, valid_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    originallabels = []
    predictedlabels = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            originallabels.extend(labels.cpu().numpy())
            predictedlabels.extend(predicted.cpu().numpy())

    val_loss /= len(valid_loader)
    accuracy = correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

    result_file = "PT_CVIP_2024.txt"
    acc = calculate_acc(originallabels, predictedlabels, result_file)
    print("Accuracy: ", acc)
    return val_loss, accuracy

# Function to plot training vs validation metrics and save the plot
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_metrics_to_csv(data, filename='metrics_data.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy'])
        for i in range(len(data['train_losses'])):
            writer.writerow([i+1, data['train_losses'][i], data['valid_losses'][i], data['train_accuracies'][i], data['valid_accuracies'][i]])

def load_metrics_from_csv(filename='metrics_data.csv'):
    if os.path.exists(filename):
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            data = {'train_losses': [], 'valid_losses': [], 'train_accuracies': [], 'valid_accuracies': []}
            for row in reader:
                data['train_losses'].append(float(row['train_loss']))
                data['valid_losses'].append(float(row['valid_loss']))
                data['train_accuracies'].append(float(row['train_accuracy']))
                data['valid_accuracies'].append(float(row['valid_accuracy']))
            return data
    return {'train_losses': [], 'valid_losses': [], 'train_accuracies': [], 'valid_accuracies': []}

def plot_metrics(metrics, filename='PLOT_PT_CVIP_2024.png'):
    epochs = range(1, len(metrics['train_losses']) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_losses'], 'bo-', label='Training loss')
    plt.plot(epochs, metrics['valid_losses'], 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_accuracies'], 'bo-', label='Training accuracy')
    plt.plot(epochs, metrics['valid_accuracies'], 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Main script
if __name__ == '__main__':

    #Base Directory path
    base_dir = "/workspace/Dataset/Dataset"

    # Define the directories for train, validation
    train_dir = os.path.join(base_dir, 'training')
    valid_dir = os.path.join(base_dir, 'validation')

    #Transformation Pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])  
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
    
    #Printing Training and Validation dataset sizes
    print('Train Size: ', len(train_dataset))
    print('Valid Size: ', len(valid_dataset))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNetResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train and validate the model
    num_epochs = 50
    checkpoint_file = 'CP-PT_CVIP_2024.pth'
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, checkpoint_file)
    #validate_model(model, valid_loader, criterion)