import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# 사용자 정의 모듈 임포트
from nets.resnet50_1_tinet import ResNet, Bottleneck
from nets.early_stopping import EarlyStopping

# 1. 인자 설정
parser = argparse.ArgumentParser(description='ResNet Model1 Training Optimization')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
args = parser.parse_args()

# 2. 하이퍼파라미터
BATCH_SIZE = 128
NUM_EPOCHS = 150
INITIAL_LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 6
CUSTOM_CONV_LAYER_INDEX = args.cusin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 3. 데이터 증강
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(64, padding=8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

train_dataset = datasets.ImageFolder("../tiny-imagenet-200/train", transform=transform_train)
val_dataset = datasets.ImageFolder("../tiny-imagenet-200/val", transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 4. 모델 및 도구 설정
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)

# [Label Smoothing] 이미 적용됨 (0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, 
                      weight_decay=WEIGHT_DECAY, nesterov=True)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
early_stopping = EarlyStopping(patience=15, delta=0.001)

# [Mixed Precision] GradScaler 초기화
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

        # [Mixed Precision] autocast 적용 (Forward pass)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # [Mixed Precision] Scaled 역전파 및 최적화
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
        # 검증 단계에서도 autocast를 사용하여 일관성을 유지하고 속도를 높임
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

    if epoch > 20:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"⛔ Early stopping triggered at epoch {epoch+1}")
            break

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f'best_model_cusin_{CUSTOM_CONV_LAYER_INDEX}.pth')
        print(f"🌟 Best Model Saved! (Acc: {best_acc:.2f}%)")

print(f"🏁 최고 정확도: {best_acc:.2f}%")