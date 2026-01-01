import torch
import sys

weight_path = "../weights/cifar10/cifar10_model-1_cusin-2_epoch-50.pth"
try:
    state_dict = torch.load(weight_path, map_location='cpu')
    print("Keys related to layer2.0.downsample:")
    for key in state_dict.keys():
        if "layer2.0.downsample" in key:
            print(key)
            
    print("\nKeys related to layer1.0.downsample:")
    for key in state_dict.keys():
        if "layer1.0.downsample" in key:
            print(key)

except Exception as e:
    print(e)
