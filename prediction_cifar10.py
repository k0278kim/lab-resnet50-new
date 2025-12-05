"""This script is used to test the accuracy of the ResNet50 model on the CIFAR-10 test dataset."""

import torch
from nets.resnet50_1_imagenet import ResNet1_imagenet, Bottleneck1_imagenet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

BATCH_SIZE = 64
CUSTOM_CONV_LAYER_INDEX = 4
NUM_WORKERS = 4
# TODO: Update this path to your trained model weights
PATH = './cifar10_model-1_cusin-4_epoch-80.pth'
NUM_SAMPLES = 1  # Set to None to use the full dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model initialization for CIFAR-10 (10 classes)
model = ResNet1_imagenet(Bottleneck1_imagenet, [3, 4, 6, 3], custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX, num_classes=10)

# Load weights if available
try:
    model.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))
    print(f"Loaded weights from {PATH}")
except FileNotFoundError:
    print(f"Warning: Weight file not found at {PATH}. Running with random weights for testing code structure.")

model = model.to(device)
model.eval()

# CIFAR-10 Normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 Test Dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

if NUM_SAMPLES is not None:
    # Use only the first NUM_SAMPLES
    indices = list(range(min(NUM_SAMPLES, len(test_dataset))))
    test_dataset = Subset(test_dataset, indices)
    print(f"Using a subset of {len(test_dataset)} samples.")

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Test Accuracy Measurement
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
