import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.resnet50 import ResNet, Bottleneck
from nets.early_stopping import EarlyStopping

# 하이퍼파라미터
BATCH_SIZE = 512
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "./resnet50-mnist.pth"
NUM_WORKERS = 0
CUSTOM_CONV_LAYER_INDEX = 1