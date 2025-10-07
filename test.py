import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm  # ✅ 추가

# ================================
# 1. 환경 설정
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
num_workers = 8
epochs = 90
lr = 0.1

# ================================
# 2. 데이터셋 정의
# ================================
data_dir = "../yoon/imagenet"  # train/, val/ 포함된 경로

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, pin_memory=True)


# ================================
# 3. 모델 정의
# ================================
model = models.resnet50(weights=None, num_classes=1000)  # 사전학습 X
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# ================================
# 4. 학습 루프 (tqdm 적용)
# ================================
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })

    return running_loss / len(train_loader), 100.*correct/total



# ================================
# 5. 실행
# ================================
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(epoch)
    scheduler.step()

    print(f"Epoch {epoch:03d} | "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | ")
