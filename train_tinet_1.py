import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.resnet50_1_imagenet import ResNet1_imagenet, Bottleneck1_imagenet
from nets.early_stopping import EarlyStopping
import argparse

parser = argparse.ArgumentParser(description='ResNet Model2 Training')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
args = parser.parse_args()

# 하이퍼파라미터
BATCH_SIZE = 64
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
NUM_WORKERS = 6
CUSTOM_CONV_LAYER_INDEX = args.cusin

# 조기 종료 조건 초기화
early_stopping = EarlyStopping(patience=5, delta=0.001)

def eval_acc(net, loader, cuda):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            if cuda:
                images = images.cuda()
                targets = targets.cuda()
            outputs = net(images)
            pred = outputs.argmax(dim=1)
            correct += (pred == targets).sum()
            total += targets.size(0)
    return 100.0 * correct / total

def fit_one_epoch(net, softmaxloss, epoch, epoch_size, epoch_size_val, gen, gen_test, Epoch, cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    acc_train_mode = eval_acc(net, gen, cuda)
    net.eval()
    acc_eval_mode = eval_acc(net, gen, cuda)

    with torch.no_grad():
        with tqdm(total=epoch_size_val, desc='Epoch{}/{}'.format(epoch + 1, Epoch), postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen_test):
                images, targets = batch[0], batch[1]
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()
                outputs = net(images)
                _, id = torch.max(outputs.data, 1)
                test_correct += torch.sum(id == targets.data)
                pbar.set_postfix(**{'test AP': float(100 * test_correct / len(test_dataset))})
                pbar.update(1)

    torch.save(net.state_dict(), 'logs/Epoch{}-Total_Loss{}.pth'.format((epoch + 1), (total_loss / ((iteration + 1)))))

    print(f"[Test1] Train-set acc with net.train(): {acc_train_mode:.2f}%")
    print(f"[Test2] Train-set acc with net.eval(): {acc_eval_mode:.2f}%")

if __name__ == '__main__':
    cuda = True
    pre_train = False
    CosineLR = True

    lr = 1e-3
    Batch_size = 512
    Init_Epoch = 0
    Fin_Epoch = 5

    model = ResNet1_imagenet(Bottleneck1_imagenet, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_dataset = datasets.ImageFolder("../tiny-imagenet-200/train", transform=transforms.ToTensor())
    val_dataset = datasets.ImageFolder("../tiny-imagenet-200/val", transform=transforms.ToTensor())

    gen = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    gen_test = DataLoader(dataset=test_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    epoch_size = len(gen)
    epoch_size_val = len(gen_test)

    softmax_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if CosineLR:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-10)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    
    for epoch in range(Init_Epoch, Fin_Epoch):
        fit_one_epoch(net=model, softmaxloss=softmax_loss, epoch=epoch, epoch_size=epoch_size,
                    epoch_size_val=epoch_size_val, gen=gen, gen_test=gen_test, Epoch=Fin_Epoch, cuda=cuda)
        lr_scheduler.step()


# 학습 루프
# for epoch in range(NUM_EPOCHS):
#     model.train()
#     running_loss = 0.0

#     pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
#     for images, labels in pbar:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         pbar.set_postfix({'loss': f"{loss.item():.4f}"})

#     avg_loss = running_loss / len(train_loader)
#     print(f"✅ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

#     # 조기 종료 체크 (여기선 train_loss 기반이지만 val_loss가 있으면 교체 가능)
#     early_stopping(avg_loss)
#     if early_stopping.early_stop:
#         print(f"⛔ Early stopping at epoch {epoch+1}")
#         torch.save(model.state_dict(), f'tinet_model-1_cusin-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')
#         break
#     elif epoch + 1 == NUM_EPOCHS:
#         torch.save(model.state_dict(), f'tinet_model-1_cusin-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')

# # 테스트 정확도 측정
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in tqdm(val_loader, desc="Testing"):
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)

# accuracy = 100 * correct / total
# print(f"✅ Test Accuracy: {accuracy:.2f}%")
