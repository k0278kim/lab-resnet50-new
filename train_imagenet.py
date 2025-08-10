import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.resnet50 import ResNet, Bottleneck
from nets.resnet50_2 import ResNet2, Bottleneck2
from nets.resnet50_2_imagenet import ResNet2_imagenet, Bottleneck2_imagenet
from nets.early_stopping import EarlyStopping
import torchvision.transforms as transforms

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 하이퍼파라미터
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "./resnet50-mnist.pth"
NUM_WORKERS = 0
CUSTOM_CONV_LAYER_INDEX = 1

# 모델 초기화
model = ResNet2_imagenet(Bottleneck2_imagenet, [3, 4, 6, 3], num_classes=1000, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX)
model = model.to(device)

# 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
        torch.save(model.state_dict(), f'resnet-model_imagenet-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')
        break
    elif epoch + 1 == NUM_EPOCHS:
        torch.save(model.state_dict(), f'resnet-model_imagenet-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')