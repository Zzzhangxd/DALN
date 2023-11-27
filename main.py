import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from DALN import DomainAdaptationModel, FeatureExtractor, Classifier
from data_loader import load_training, load_testing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import time

# 参数设置
batch_size = 36
root_path = 'dataset'
source_dir = 'data20230531'  # 源数据集目录
target_dir = 'data20230601'  # 目标数据集目录
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
best_accuracy = 0  # 用于保存最佳准确率
best_preds = None
best_targets = None

def train(model, train_loader_s, train_loader_t, optimizer, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_idx, (data_s, data_t) in enumerate(zip(train_loader_s, train_loader_t)):
        inputs_s, labels_s = data_s
        inputs_t, _ = data_t

        optimizer.zero_grad()
        loss = model(inputs_s, inputs_t, labels_s)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs_s)}/{len(train_loader_s.dataset)} '
                  f'({100. * batch_idx / len(train_loader_s):.0f}%)]\tLoss: {loss.item():.6f}')

    end_time = time.time()
    epoch_loss = total_loss / len(train_loader_s)
    print(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s')

def test(model, test_loader):
    global best_accuracy, best_preds, best_targets
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            output = model.classifier(model.feature_extractor(data))
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    end_time = time.time()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%), Time: {end_time - start_time:.2f}s\n')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_preds = pred.cpu().numpy()
        best_targets = target.cpu().numpy()
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Saved best model with accuracy: {accuracy:.2f}%')

    return test_loss, accuracy

def plot_confusion_matrix(preds, targets):
    conf_mat = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d', ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def main():
    train_loader_s = load_training(root_path, source_dir, batch_size, kwargs)
    train_loader_t = load_training(root_path, target_dir, batch_size, kwargs)
    test_loader = load_testing(root_path, target_dir, batch_size, kwargs)

    feature_extractor = FeatureExtractor()
    classifier = Classifier()
    model = DomainAdaptationModel(feature_extractor, classifier)

    optimizer_params = [
        {'params': feature_extractor.parameters(), 'lr': 1e-3},
        {'params': classifier.parameters(), 'lr': 1e-2}
    ]

    optimizer = optim.SGD(optimizer_params, momentum=0.9, weight_decay=1e-4)
    
    # 使用StepLR调度器
    scheduler = StepLR(optimizer, step_size=1, gamma=0.3)

    epochs = 10  # 设置训练轮数
    for epoch in range(1, epochs + 1):
        train(model, train_loader_s, train_loader_t, optimizer, epoch)
        test_loss, _ = test(model, test_loader)
        scheduler.step()  # 更新学习率

    # 训练完成后，如果有最佳预测结果，则绘制混淆矩阵
    if best_preds is not None and best_targets is not None:
        plot_confusion_matrix(best_preds, best_targets)

if __name__ == '__main__':
    main()
