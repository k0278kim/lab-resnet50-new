import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

# 사용자 정의 모듈 임포트
# 환경에 따라 import 경로는 수정이 필요할 수 있습니다.
from nets.resnet50_1_tinet import ResNet_pure, Bottleneck_pure

# 1. 인자 설정
parser = argparse.ArgumentParser(description='ResNet-50 CIFAR-10 Optimized Training')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
args = parser.parse_args()

# 2. 하이퍼파라미터 (PPTX 표준 레시피 반영)
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
INITIAL_LR = args.lr
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1
CUSTOM_CONV_LAYER_INDEX = args.cusin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 3. 데이터 로드 (PPTX 슬라이드 12 표준 설정)
# CIFAR-10 전용 정규화 및 증강 기법 적용
stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=stats[1])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=stats[1])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 4. 모델 설정
model = ResNet_pure(Bottleneck_pure, [3, 4, 6, 3], num_classes=10, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)

# CIFAR-10 입력(32x32)에 최적화된 초기 레이어 수정
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.bn1 = nn.Identity() # 사용자 요구사항 유지 (Identity)

model.to(device)

# 가중치 로드 설정
checkpoint_path = f'best_cifar10_cusin_{CUSTOM_CONV_LAYER_INDEX}.pth'
if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
else:
    print("🆕 Starting training from scratch.")

# 5. 손실 함수 및 옵티마이저 (PPTX 레시피)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, 
                      weight_decay=WEIGHT_DECAY)

# 스케줄러: Cosine Annealing (Option B)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = torch.cuda.amp.GradScaler() # Mixed Precision 학습 지원

# 6. 학습 루프
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # --- [TRAIN PHASE] ---
    model.train()
    train_loss = 0.0
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
        
        train_loss += loss.item()
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
    print(f"📊 Result: Val Loss = {avg_val_loss:.4f} | Val Acc = {val_acc:.2f}%")

    # 스케줄러 업데이트
    scheduler.step()

    # 모델 저장 (Best Accuracy 기준)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), checkpoint_path)
        print(f"🌟 Best Model Saved! (Acc: {best_acc:.2f}%)")

print(f"🏁 Final Best Accuracy: {best_acc:.2f}%")