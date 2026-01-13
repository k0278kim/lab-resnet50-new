import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

# 사용자 정의 모듈 임포트
from nets.resnet50_2_tinet import ResNet, Bottleneck
from nets.early_stopping import EarlyStopping

# 1. 인자 설정
parser = argparse.ArgumentParser(description='ResNet Model2 Training Optimization')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
args = parser.parse_args()

# 2. 하이퍼파라미터 (SOTA Recipe 기반)
BATCH_SIZE = 128          # BN 안정성을 위해 128 권장
NUM_EPOCHS = 120          # Scratch 학습을 위한 충분한 기간
INITIAL_LR = 0.1          # SGD 표준 시작 학습률
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 6
CUSTOM_CONV_LAYER_INDEX = args.cusin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 3. 데이터 증강 (Data Augmentation) - Tiny-ImageNet 맞춤형
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(64, padding=8), # 64x64 유지
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

train_dataset = datasets.MNIST(root='data/', train=True,                            #convert train to False
                                   transform=transforms.ToTensor(), download=False)
test_da= datasets.MNIST(root='data/', train=False,
                                  transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 4. 모델, 손실함수, 최적화 도구 설정
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)

# Label Smoothing은 클래스 간 경계를 부드럽게 하여 성능을 높입니다.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Nesterov Momentum SGD 적용
optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, 
                      weight_decay=WEIGHT_DECAY, nesterov=True)

# CosineAnnealing 스케줄러 (학습률이 부드럽게 감소)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# 조기 종료 조건 (Patience를 15로 늘려 초기 불안정기를 견디게 함)
early_stopping = EarlyStopping(patience=15, delta=0.001)

# 5. 학습 루프
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # --- [TRAIN PHASE] ---
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.5f}"})

    # --- [VALIDATION PHASE] ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()  # ◀ 오타 수정: 누적 합계로 변경
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    avg_val_loss = val_loss / len(test_loader)
    val_acc = 100 * correct / total
    print(f"📊 Epoch {epoch+1} 결과: test Loss = {avg_val_loss:.4f} | test Acc = {val_acc:.2f}%")

    # 스케줄러 단계 진행
    scheduler.step()

    # --- [EARLY STOPPING & SAVING] ---
    # 초기 20에폭 동안은 모델이 자리를 잡는 시기이므로 조기 종료를 유예합니다.
    if epoch > 20:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"⛔ Early stopping triggered at epoch {epoch+1}")
            break

    # 베스트 모델 저장
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f'best_model_cusin_{CUSTOM_CONV_LAYER_INDEX}.pth')
        print(f"🌟 Best Model Saved! (Acc: {best_acc:.2f}%)")

print(f"🏁 학습 완료! 최고 정확도: {best_acc:.2f}%")