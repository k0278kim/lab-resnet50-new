import torch
from nets.resnet50_2_imagenet import ResNet2_imagenet, Bottleneck2_imagenet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import cv2
import time
from tqdm import tqdm

# Load model
# Path to the pretrained model
PATH = './resnet-model_imagenet-1_epoch-1.pth'
# Ask user for batch size
# Batch_Size = int(input('The number of handwritten font images predicted each times：'))
Batch_Size = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet2_imagenet(Bottleneck2_imagenet, [3, 4, 6, 3], num_classes=1000, custom_conv_layer_index=1)
model.load_state_dict(torch.load(PATH, map_location=device))
model = model.to(device)
model.eval()

#Load test dataset
test_dataset = datasets.MNIST(root='/data/imagenet/val', train=False,
                                    transform=transforms.ToTensor(), download=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=Batch_Size, shuffle=False)

# Accuracy evaluation
correct = 0
total = 0
n = 0
total_n = 10000
with torch.no_grad():
    pbar = tqdm(test_loader, total=len(test_loader), desc="Testing")
    for images, labels in pbar:
        if (n < total_n):
            # print(f"\nBatch {i}: loaded")
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            pbar.set_postfix({'Accuracy (%)': f"{accuracy:.2f}"})
            n += 1
        else:
            break


# accuracy = 100 * correct / total
print(f"✅ Accuracy on the MNIST test set: {accuracy:.2f}%")