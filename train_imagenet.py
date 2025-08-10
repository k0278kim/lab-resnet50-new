import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.resnet50_2_imagenet import ResNet2_imagenet, Bottleneck2_imagenet
from nets.early_stopping import EarlyStopping

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 하이퍼파라미터
BATCH_SIZE = 64
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "./resnet50-mnist.pth"
NUM_WORKERS = 2
CUSTOM_CONV_LAYER_INDEX = 1

# 모델 초기화
model = ResNet2_imagenet(Bottleneck2_imagenet, [3, 4, 6, 3], num_classes=1000, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX)
model = model.to(device)

# 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# AMP용 GradScaler 초기화
scaler = torch.cuda.amp.GradScaler()

# 학습 데이터셋
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder("/data/imagenet/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)

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

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # AMP 스케일러로 backward와 step 처리
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    # 조기 종료 체크 (여기선 train_loss 기반)
    early_stopping(avg_loss)
    if early_stopping.early_stop:
        print(f"⛔ Early stopping at epoch {epoch+1}")
        torch.save(model.state_dict(), f'resnet-model_imagenet-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')
        break
    elif epoch + 1 == NUM_EPOCHS:
        torch.save(model.state_dict(), f'resnet-model_imagenet-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')
