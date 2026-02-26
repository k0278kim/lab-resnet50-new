import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

# 사용자 정의 모듈 임포트
from nets.resnet50_1_tinet import ResNet_pure, Bottleneck_pure
from nets.early_stopping import EarlyStopping
from tiny_imagenet_dataset import load_data  # 기존에 검증된 데이터 로더 필수 활용

# 1. 인자 설정
parser = argparse.ArgumentParser(description='ResNet Model1 Training Optimization')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
args = parser.parse_args()

# 2. 하이퍼파라미터 (기존 설정 유지)
BATCH_SIZE = 128
NUM_EPOCHS = args.epochs
INITIAL_LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 6
CUSTOM_CONV_LAYER_INDEX = args.cusin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 3. 데이터 로드 (tiny_imagenet_dataset.py의 load_data 활용)
# ImageFolder의 구조적 한계를 극복하기 위해 기존에 사용하시던 로직을 호출합니다.
train_dir = "../tiny-imagenet-200/train"
val_dir = "../tiny-imagenet-200/val"

# 기존에 구현된 load_data는 Tiny ImageNet의 특수 구조(val_annotations.txt 등)를 처리합니다.
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

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS, 
    pin_memory=True
)
val_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True
)

# 4. 모델 및 도구 설정
model = ResNet_pure(Bottleneck_pure, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.bn1 = nn.Identity()

model.to(device)

checkpoint_path = f'best_cifar10_cusin_{CUSTOM_CONV_LAYER_INDEX}.pth'

if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    # map_location은 GPU/CPU 환경이 달라도 안전하게 로드하기 위해 사용합니다.
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Weights loaded successfully. Resuming training...")
else:
    print("🆕 No checkpoint found. Starting from scratch.")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, 
                      weight_decay=WEIGHT_DECAY, nesterov=True)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
early_stopping = EarlyStopping(patience=15, delta=0.001)
scaler = torch.cuda.amp.GradScaler()

# 5. 학습 루프
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # --- [TRAIN PHASE] ---
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.5f}"})

    # --- [VALIDATION PHASE] ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(f"📊 결과: Val Loss = {avg_val_loss:.4f} | Val Acc = {val_acc:.2f}%")

    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f'best_cifar10_cusin_{CUSTOM_CONV_LAYER_INDEX}.pth')
        print(f"🌟 Best Model Saved! (Acc: {best_acc:.2f}%)")

print(f"🏁 최고 정확도: {best_acc:.2f}%")