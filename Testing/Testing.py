from tensorflow.keras.models import load_model
import numpy as np
from Eval_metrics_gen_excel import save_predictions_to_excel,generate_metrics_report
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import platform
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F
from functools import partial
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2
import pywt
import torch.autograd


#Encoder
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
        self.oga5 = ODConv2d(512, 512, 3)

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
            

# Decoder
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
    def __init__(self, pretrained=True, encoder_depth=34, num_in_channels=4, num_classes=38):
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

    def __init__(self, num_classes = 38, drop_ratio = 0.5, number_of_channels=13):

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
        #x = self.bn(x)
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

        # Weight parameters for dynamic kernels
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

    # Update the temperature for attention scaling
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


# UNetResNet18
class UNetResNet18(nn.Module):
    def __init__(self):
        super(UNetResNet18, self).__init__()
        self.encoder = Encoder(3, 10)
        self.decoder = DeCoder(3)
        self.resnet = Classifier(num_classes = 10, number_of_channels=3)
        

    def forward(self, x):
        enc_out = self.encoder(x)
        
        enc_outs = list(enc_out)
        to_grayscale = ToGrayscale()
        swt_image = apply_swt(to_grayscale(x))
        swt_images_resized = F.interpolate(swt_image, size=enc_outs[-1].shape[2:], mode='nearest')
        
        dwt_image = apply_dwt(to_grayscale(x))
        dwt_images_resized = F.interpolate(dwt_image, size=enc_outs[-2].shape[2:], mode='nearest')
        
        to_grayscale_resized2 = F.interpolate(to_grayscale(x), size=enc_outs[-3].shape[2:], mode='nearest')
        
        enc_outs[-1] = torch.cat([enc_outs[-1], swt_images_resized], dim=1)    # Concatenate SWT components (4 channels) to (512, 8, 8)
        enc_outs[-2] = torch.cat([enc_outs[-2], dwt_images_resized], dim=1)    # Concatenate DWT components (2 channels) to (256, 16, 16)
        enc_outs[-3] = torch.cat([enc_outs[-3], to_grayscale_resized2], dim=1) # Concatenate GrayScale
        
        dec_outs = self.decoder(enc_outs)
        
        resnet_out = self.resnet(dec_outs[-1])
        return resnet_out


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

#modify according to the model type being used
#these data loading functions are specific to VGG16, modify accordingly 
def load_and_preprocess_image(full_path, target_size):
    img = load_img(full_path, target_size=target_size)
    img_array = img_to_array(img)
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img
def get_data(excel_path, base_dir, image_size=(32, 32)):
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['image_path'])
    if platform.system() == 'Windows':
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('/', os.sep))
    else:
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('\\', os.sep))
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    X = np.array([load_and_preprocess_image(os.path.join(base_dir, path), image_size) for path in df['image_path'].values])
    y = df[class_columns].values
    return X, y, df
def load_test_data(test_dir, image_size=(224, 224)):
    image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.lower().endswith(('jpg'))]
    X_test = np.array([load_and_preprocess_image(path, image_size) for path in image_paths])
    print(X_test)
    return X_test, image_paths

# Function to load a checkpoint
def load_checkpoint(model, optimizer, filename='CP-PT_CVIP_2024.pth'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

def save_predictions_to_excel(image_paths, y_pred, output_path):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    y_pred_classes = np.argmax(y_pred, axis=1)
    predicted_class_names = [class_columns[i] for i in y_pred_classes]
    print(predicted_class_names)
    df_prob = pd.DataFrame(y_pred, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)
    df_class = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
    df_merged = pd.merge(df_prob, df_class, on='image_path')
    df_merged.to_excel(output_path, index=False)
criterion = nn.CrossEntropyLoss()


class CustomeDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir=data_dir
        self.transform = transform
        self.file_names = os.listdir(data_dir)
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self,idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_dir,file_name)
        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)
        label = file_path.split("/")[-1]
        return image,label,file_path

#Test-data directory path
test_dir="Path_To_Test_Directory"

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
    ])  

test_dataset = CustomeDataset(data_dir=test_dir, transform=transform)

# DataLoaders
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = UNetResNet18().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
checkpoint_file ="CP-PT_CVIP_2024"
start_epoch = load_checkpoint(model, optimizer, checkpoint_file)
all_preds=[]
file_paths=[]
output_test_predictions="test_excelSGDAug.xlsx"
for inputs,label,file_path in test_loader:
    print(file_path)
    inputs = inputs.to(device)
    outputs = model(inputs)
    all_preds.extend(outputs.cpu().detach().numpy())
    file_paths.extend(list(file_path))

save_predictions_to_excel(file_paths,all_preds,output_test_predictions)
