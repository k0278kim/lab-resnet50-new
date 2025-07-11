import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.resnet50 import ResNet, Bottleneck
from nets.early_stopping import EarlyStopping

# 하이퍼파라미터
BATCH_SIZE = 1
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "./resnet50-mnist.pth"
NUM_WORKERS = 0
CUSTOM_CONV_LAYER_INDEX = 1

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 모델 초기화
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)

# 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 데이터셋
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# 테스트 데이터셋
test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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

    # 모델 저장
    torch.save(model.state_dict(), f'resnet-model_cusin-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')

    # 조기 종료 체크 (여기선 train_loss 기반이지만 val_loss가 있으면 교체 가능)
    early_stopping(avg_loss)
    if early_stopping.early_stop:
        print(f"⛔ Early stopping at epoch {epoch+1}")
        break
    elif epoch + 1 == NUM_EPOCHS:
        torch.save(model.state_dict(), f'resnet-model_cusin-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')

# 테스트 정확도 측정
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
