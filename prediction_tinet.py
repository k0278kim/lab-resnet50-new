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

test_dataset = datasets.ImageFolder("../tiny-imagenet-200/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# 테스트 정확도 측정
model.eval()
top1_correct = 0
top5_correct = 0
total = 0

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

n_dat = 101
with torch.no_grad():
    pbar = tqdm(test_loader, total=n_dat, desc="Testing")
    idx = 0
    for images, labels in pbar:
        if (idx == n_dat):
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        # 정의하신 accuracy 함수 활용 (batch_size=1이므로 결과는 각 0 또는 100)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        # 누적 계산 (백분율을 다시 개수로 변환)
        batch_size = labels.size(0)
        top1_correct += acc1.item() * batch_size / 100
        top5_correct += acc5.item() * batch_size / 100
        total += batch_size
        
        idx += 1
        
        if idx % 10 == 0: # 진행 상황 업데이트 주기 조절
            pbar.set_postfix({
                'Top-1 (%)': f"{100 * top1_correct / total:.2f}",
                'Top-5 (%)': f"{100 * top5_correct / total:.2f}"
            })

# 최종 결과 출력
final_top1 = 100 * top1_correct / total
final_top5 = 100 * top5_correct / total

print(f"\n✅ Final Results (Samples: {total})")
print(f"⭐ Top-1 Accuracy: {final_top1:.2f}%")
print(f"⭐ Top-5 Accuracy: {final_top5:.2f}%")