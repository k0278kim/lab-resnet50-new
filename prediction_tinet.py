"""This script is used to test the accuracy of the ResNet50 model on the MNIST test dataset."""

import torch
from nets.resnet50_1_imagenet import ResNet1_imagenet, Bottleneck1_imagenet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import cv2
import time
from tqdm import tqdm

BATCH_SIZE = 1
CUSTOM_CONV_LAYER_INDEX = 1
NUM_WORKERS = 6
PATHS = ['../weights/tinet/tinet_model-1_cusin-1_epoch-61.pth', '../weights/tinet/tinet_model-1_cusin-2_epoch-63.pth', '../weights/tinet/tinet_model-1_cusin-3_epoch-66.pth', '../weights/tinet/tinet_model-1_cusin-4_epoch-73.pth']
PATH = PATHS[CUSTOM_CONV_LAYER_INDEX - 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet1_imagenet(Bottleneck1_imagenet, [3, 4, 6, 3],custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX, num_classes=200)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'), weights_only=True))
model = model.to(device)
model.eval()

transform = transforms.ToTensor()
train_dataset = datasets.ImageFolder("../tiny-imagenet-200/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

val_dataset = datasets.ImageFolder("../tiny-imagenet-200/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

def accuracy(output, target, topk=(1, 5)):
    """Top-k 정확도 계산 함수"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 상위 maxk개의 예측 인덱스를 가져옴
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 차원 변경 (maxk, batch_size)

        # 정답 레이블과 비교하여 맞았는지 확인 (정답을 동일한 형상으로 확장)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # 상위 k개 안에 정답이 포함된 개수를 합산
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res # [Top-1 Accuracy, Top-5 Accuracy]

model.eval()
top1_cnt = 0
top5_cnt = 0
total_samples = 0
n_dat = 100 # 테스트할 샘플 수

with torch.no_grad():
    pbar = tqdm(enumerate(val_loader), total=n_dat, desc="Testing")
    for i, (images, labels) in pbar:
        if i >= n_dat:
            break
        
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Top-5 예측 인덱스 추출
        _, pred5 = outputs.topk(5, 1, True, True)
        
        # Top-1 판정
        top1_cnt += (pred5[:, 0] == labels).sum().item()
        # Top-5 판정 (5개 예측값 중 정답이 있는지 확인)
        top5_cnt += (pred5 == labels.view(-1, 1)).any(dim=1).sum().item()
        
        total_samples += labels.size(0)
        
        pbar.set_postfix({
            'Top-1': f"{100 * top1_cnt / total_samples:.2f}%",
            'Top-5': f"{100 * top5_cnt / total_samples:.2f}%"
        })

print(f"\n⭐ 최종 Top-1: {100 * top1_cnt / total_samples:.2f}%")
print(f"⭐ 최종 Top-5: {100 * top5_cnt / total_samples:.2f}%")