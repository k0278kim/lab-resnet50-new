import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.resnet50_1_imagenet import ResNet1_imagenet, Bottleneck1_imagenet
from nets.resnet50_2_tinet import ResNet, Bottleneck
from nets.early_stopping import EarlyStopping
import os
import argparse

parser = argparse.ArgumentParser(description='ResNet Model Training')
parser.add_argument('--model', type=int, default=1, help='model number')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
args = parser.parse_args()

# 하이퍼파라미터
BATCH_SIZE = 128  # 64 -> 128 (ResNet은 배치가 좀 커도 됨)
NUM_EPOCHS = 100  # 80 -> 100 (스케줄링을 위해 조금 늘림)
LEARNING_RATE = 0.1 # Adam 대신 SGD를 쓸 경우 0.1부터 시작하는 게 국룰
NUM_WORKERS = 4
CUSTOM_CONV_LAYER_INDEX = args.cusin
MODEL_NUMBER = args.model

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# [수정 1] 데이터 전처리 강화 (필수!)
# CIFAR-10 평균/표준편차 정규화 및 증강
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     # 이미지를 랜덤하게 자름 (위치 정보 변화)
    transforms.RandomHorizontalFlip(),        # 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize(*stats)              # 정규화 (학습 안정성)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)              # 테스트 때도 정규화는 필수
])

# 데이터셋 로드
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 모델 초기화
# [주의] ResNet1_imagenet 내부의 conv1이 kernel_size=3, stride=1, padding=1 인지 꼭 확인하세요!
# 만약 ImageNet용 그대로(7x7)라면 성능 안 나옵니다.
model = ResNet1_imagenet(Bottleneck1_imagenet, [3, 4, 6, 3], num_classes=10, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)
# [수정 2] Optimizer 변경 및 Scheduler 추가
criterion = nn.CrossEntropyLoss()
# ResNet + CIFAR10 조합은 Adam보다 SGD+Momentum이 최고 성능을 냅니다.
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

# 학습률 스케줄러: 50, 75 epoch에서 학습률을 1/10로 줄임
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

# 조기 종료 (Test Accuracy 기준이 더 안전함. 여기선 일단 뺍니다. 끝까지 돌리는 게 나음)
# early_stopping = EarlyStopping(patience=10, delta=0.001) 

# 학습 루프
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] LR={optimizer.param_groups[0]['lr']:.5f}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Train Acc 확인용
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # 스케줄러 업데이트 (에폭 끝날 때마다)
    scheduler.step()

    train_acc = 100 * correct_train / total_train
    avg_loss = running_loss / len(train_loader)
    
    # 테스트 정확도 측정 (매 에폭마다 확인)
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_acc = 100 * correct_test / total_test
    
    print(f"✅ Epoch {epoch+1}: Loss={avg_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")

    # 최고 성능 모델 저장
    if test_acc > best_acc:
        best_acc = test_acc
        print(f"🔥 Best Acc Updated: {best_acc:.2f}% -> Saving Model...")
        torch.save(model.state_dict(), f'cifar10_model-{MODEL_NUMBER}_cusin-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')

print(f"🚀 Final Best Test Accuracy: {best_acc:.2f}%")