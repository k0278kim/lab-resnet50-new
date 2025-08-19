import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.resnet50_1_imagenet import ResNet1_imagenet, Bottleneck1_imagenet
from nets.early_stopping import EarlyStopping
import torchvision.transforms as transforms
import os

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 하이퍼파라미터
BATCH_SIZE = 1024
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "./resnet50-mnist.pth"
RESUME_PATH = "checkpoint.pth"  # 이전 학습 모델 경로
NUM_WORKERS = 4
CUSTOM_CONV_LAYER_INDEX = 1

# 모델 초기화
model = ResNet1_imagenet(Bottleneck1_imagenet, [3, 4, 6, 3], num_classes=1000, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX)
model = model.to(device)

# 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_epoch = 0
if os.path.exists(RESUME_PATH):
    print(f"🔄 Loading checkpoint from {RESUME_PATH}...")
    checkpoint = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"✅ Resumed training from epoch {start_epoch}")

# 학습 데이터셋
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder("/data/imagenet/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# 테스트 데이터셋
# test_dataset = datasets.ImageFolder("/data/imagenet/val", transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 조기 종료 조건 초기화
early_stopping = EarlyStopping(patience=5, delta=0.001)

# 학습 루프
for epoch in range(NUM_EPOCHS):
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
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    # 조기 종료 체크 (여기선 train_loss 기반이지만 val_loss가 있으면 교체 가능)
    early_stopping(avg_loss)
    if early_stopping.early_stop:
        print(f"⛔ Early stopping at epoch {epoch+1}")
        torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, "checkpoint.pth")

        break
    elif epoch + 1 == NUM_EPOCHS:
        torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, "checkpoint.pth")
