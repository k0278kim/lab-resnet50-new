import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes   # Load the shared library  
cal = ctypes.cdll.LoadLibrary("./calculator.so")

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias as trainable parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x_padded, in_height, in_width):
        return self.custom_conv2d(x_padded, in_height, in_width, self.weight, self.bias, self.stride, self.padding)

    def custom_conv2d(self, input_padded, in_height, in_width, weight, bias=None, stride=1, padding=0):
        batch_size, in_channels, _, _ = input_padded.shape
        # batch_size, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape

        # Padding
        # input_padded = F.pad(input, (padding, padding, padding, padding))   #torch.Size([5, 1, 32, 32]), input_padded.cpu().numpy().shape = (5, 1, 32, 32)
  
        # Compute output dimensions
        out_height = (in_height + 2 * padding - kernel_height) // stride + 1
        out_width = (in_width + 2 * padding - kernel_width) // stride + 1

        # Initialize output tensor
        output = torch.zeros((batch_size, out_channels, out_height, out_width), device=input_padded.device)

        # Convolution operation
        for b in range(batch_size):  # Iterate over batch
            for o in range(out_channels):  # Iterate over output channels
                for i in range(in_channels):  # Iterate over input channels
                    for h in range(out_height):
                        for w in range(out_width): 
                            h_start, w_start = h * stride, w * stride
                            region = input_padded[b, i, h_start:h_start + kernel_height, w_start:w_start + kernel_width]
                            # output[b, o, h, w] += torch.sum(region * weight[o, i])  # Element-wise multiplication
                            #Use .copy() - Fix: Make it Contiguous First
                            region_np = np.ascontiguousarray(region.detach().cpu().numpy(), dtype=np.float32) # change arr1 for ctypes object
                            weight_np = np.ascontiguousarray(weight[o, i].detach().cpu().numpy(), dtype=np.float32)             # change arr2 for ctypes object
                            # weight_np = np.ascontiguousarray(weight_tensor.detach().cpu().numpy(), dtype=np.float32) # change arr2 for ctypes object
                            region_ctypes = region_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                            weight_ctypes = weight_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                            output[b, o, h, w] = cal.matrix_product(region_ctypes, weight_ctypes, kernel_height, kernel_width)
 
                            torch_result = torch.sum(region * weight[o, i]).item()  # Element-wise multiplication
                            if not np.isclose(output[b, o, h, w].item(), torch_result, rtol=1e-3, atol=1e-5):
                                print(f"Mismatch at (b={b}, o={o}, i={i}, h={h}, w={w})")
                                print(f"  C result     = {output[b, o, h, w]}")
                                print(f"  Torch result = {torch_result}")

                # Add bias after processing all input channels
                if bias is not None:
                    output[b, o, :, :] += bias[o]

        return output


class Bottleneck(nn.Module):
    '''
    Contains three types of convolutional layers
    conv1-Number of compression channels
    conv2-Extract features
    conv3-extended number of channels
    This structure can better extract features, deepen the network, and reduce the number of network parameters。
    inplanes - in_channels
    planes = out_channels
    '''

    expansion = 4
    skip_expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Initialize weights using the Kaiming He initialization as in nn.conv2d
        # self.conv1_weight = kaiming_he_init((planes, inplanes, 1, 1))           # 1x1 conv
        # self.conv2_weight = kaiming_he_init((planes, planes, 3, 3))             # 3x3 conv
        # self.conv3_weight = kaiming_he_init((planes * 4, planes, 1, 1))         # 1x1 conv
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        # self.conv1 = CustomConv2D(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = CustomConv2D(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.conv3 = CustomConv2D(inplanes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


        
    def forward(self, x):
        '''
        This block implements the residual block structure

        ResNet50 has two basic blocks，naming Conv Block & Identity Block，resnet50 uses these two structures stacked together。
        The biggest difference between them is whether there is convolution on the residual edge。

        Identity Blockis the normal residual structure，There is no convolution on the residual side，and the input is directly added to the output；
        The residual edge of Conv Block adds convolution operation and BN operation (batch normalization)，Its function is to change the number of channels 
        of the convolution operation step，Achieve the effect of changing the network dimension。

        也就是说
        Identity Block input dimension and output dimension are the same，Can be connected in series，To deepen the network；
        Conv Block input and output dimensions are different，Therefore, it cannot be connected in series，Its function is to change the dimension of the network。
        :param
        x:输入数据
        :return:
        out:网络输出结果
        '''
        residual = x            # residual =identity

        # Convert PyTorch tensor to NumPy for processing
        # if isinstance(x, torch.Tensor):
        #     x = x.numpy()
        out = self.conv1(x)
        # out = conv2d_no_library(x, self.conv1_weight, stride=1, padding=0)
        out = self.bn1(out)
        out = self.relu(out)

        print(f'out1: {np.array(out).shape}')

        out = self.conv2(out)
        # out = conv2d_no_library(out, self.conv2_weight, stride=self.stride, padding=1)
        out = self.bn2(out)
        out = self.relu(out)

        print(f'out2: {np.array(out).shape}')

        out = self.conv3(out)
        # out = conv2d_no_library(out, self.conv3_weight, stride=1, padding=0)
        out = self.bn3(out)

        print(f'out3: {np.array(out).shape}')

        if self.downsample is not None:
            residual = self.downsample(x)
            print(f'residual: {np.array(residual).shape}')

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        # -----------------------------------#
        #   Assume that the input image is 600,600,3
        # -----------------------------------#
        self.inplanes = 64
        self.skip_planes = 32
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, bias=False)                    #TODO: original conv
        # self.conv1_weight = kaiming_he_init((64, 1, 3, 3))
        # Convert input tensor to numpy array and pass it to C function
        cal.matrix_product.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        cal.matrix_product.restype = ctypes.c_float
        
        self.conv1 = CustomConv2D(1, 64, kernel_size=3, stride=1, padding=2, bias=False)                  #FIXME: custom conv
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 16, layers[0], skip_planes=32)
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        # Here you can get a shared feature layer of 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中 - Used in the classifier model
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     #if isinstance(m, nn.Conv2d):
        #     if isinstance(m, conv2d_no_library):                                                                #FIXME: custom conv
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, skip_planes, stride=1):
        '''
        Used to construct a stack of Conv Block and Identity Block
        :param block:就是上面的Bottleneck，Used to implement the most basic residual block structure in resnet50
        :param planes:Number of output channels
        :param blocks:Residual block repetition times
        :param stride:step size
        :return:
        Constructed Conv Block & Identity Block Stacked network structure

        ''' 
        downsample = None
        # -------------------------------------------------------------------#
        # When the model needs to be compressed in height and width, downsampling of the residual edge is required.
        # -------------------------------------------------------------------#

        # 边（do构建Conv Block的残差wnsample）
        if stride != 1 or self.inplanes != planes * block.expansion:# block.expansion=4
            downsample = nn.Sequential(
                nn.Conv2d(self.skip_planes, planes * block.skip_expansion, kernel_size=1, stride=stride, bias=False),    #TODO: original conv
            # CustomConv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),   #TODO: custom conv
                nn.BatchNorm2d(planes * block.skip_expansion),
            )
            self.skip_planes = skip_planes * block.skip_expansion
       
        layers = [] # For stacking Conv Block 和 Identity Block
        # Add一a layer of Conv Block
        layers.append(block(self.inplanes, planes, stride, downsample))
        # After adding, the input dimension changed，So change inplanes (input dimension)
        self.inplanes = planes * block.expansion
        # Adding blocks layer Identity Block
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        padding = 2
        x_padded = F.pad(x, (padding, padding, padding, padding))                                              #FIXME: padding outside conv
        _, _, in_height, in_width = x.shape
    
        x = self.conv1(x_padded, in_height, in_width)
        # x = CustomConv2D(x)
        #x_test = self.conv1_org(x)                            #FIXME: for comparison
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # ----------------------------------------------------------------------------#
    #   Get the feature extraction part, from conv1 to model.layer3, and finally get a 38,38,1024 feature layer
    # ----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   Get the classification part from model.layer4 to model.avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier