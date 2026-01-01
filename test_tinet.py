import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from PIL import Image
from nets.resnet50_1_imagenet import ResNet1_imagenet, Bottleneck1_imagenet
from nets.resnet50_2_tinet import ResNet, Bottleneck
import argparse

parser = argparse.ArgumentParser(description='ResNet Test')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
parser.add_argument('--model', type=int, default=1, help='model number')
args = parser.parse_args()

# Hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 4
CUSTOM_CONV_LAYER_INDEX = args.cusin
WEIGHT_PATHS = [
    [
        "../weights/tinet/tinet_model-1_cusin-1_epoch-61.pth",
        "../weights/tinet/tinet_model-1_cusin-2_epoch-63.pth",
        "../weights/tinet/tinet_model-1_cusin-3_epoch-66.pth",
        "../weights/tinet/tinet_model-1_cusin-4_epoch-73.pth"
    ],
    [
        "../weights/tinet/tinet_model-2_cusin-1_epoch-56.pth",
        "../weights/tinet/tinet_model-2_cusin-2_epoch-57.pth",
        "../weights/tinet/tinet_model-2_cusin-3_epoch-37.pth",
        "../weights/tinet/tinet_model-2_cusin-4_epoch-59.pth"
    ]
]
WEIGHT_PATH = WEIGHT_PATHS[args.model - 1][CUSTOM_CONV_LAYER_INDEX - 1]

# Custom Dataset for Tiny ImageNet Validation
class TinyImageNetValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images_dir = os.path.join(root, 'images')
        self.annotations_file = os.path.join(root, 'val_annotations.txt')
        
        # Load training dataset ID to index mapping to ensure consistency
        # Assuming typical structure: ../tiny-imagenet-200/train
        train_dir = os.path.join(os.path.dirname(root), 'train')
        if os.path.exists(train_dir):
             # We only need the class_to_idx mapping, so we don't load images
            train_ds = datasets.ImageFolder(train_dir)
            self.class_to_idx = train_ds.class_to_idx
        else:
            print(f"Warning: Train directory not found at {train_dir}. Class mapping might be incorrect if not standard.")
            # Fallback or error handling could go here
            self.class_to_idx = {} 

        self.data = []
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name, class_wnid = parts[0], parts[1]
                        if class_wnid in self.class_to_idx:
                            self.data.append((img_name, self.class_to_idx[class_wnid]))
        else:
            print(f"Error: Annotation file not found at {self.annotations_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# CUDA Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Model
if args.model == 1: 
    model = ResNet1_imagenet(Bottleneck1_imagenet, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)
else:
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)

# Load Checkpoint
if os.path.isfile(WEIGHT_PATH):
    print(f"Loading weights from '{WEIGHT_PATH}'")
    model.load_state_dict(torch.load(WEIGHT_PATH))
else:
    print(f"No weight file found at '{WEIGHT_PATH}'")
    exit()

# Test Dataset
transform = transforms.ToTensor()
# Use Custom Dataset instead of ImageFolder
val_dataset = TinyImageNetValDataset(root="../tiny-imagenet-200/val", transform=transform)

if len(val_dataset) == 0:
    print("Error: No valid data found in validation dataset.")
    exit()

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True) # Shuffle False for testing usually

# Evaluation
model.eval()
correct = 0
total = 0

print(f"Starting Evaluation on Validation Set ({len(val_dataset)} images)...")
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Final Test Accuracy: {accuracy:.2f}%")
