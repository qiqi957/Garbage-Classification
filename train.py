"""
训练脚本 - 垃圾分类模型训练（含对比实验）
"""
import os
import random
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets

import config
from sign_classifier import create_model
from utils import (
    plot_class_distribution,
    plot_confusion_matrix,
    compute_metrics,
    get_data_augmentation,
    plot_comparison_results
)


def set_seed(seed):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_data_loaders(aug_type="strong", use_weighted_sampler=True):
    """准备数据加载器"""
    train_transform, val_transform = get_data_augmentation(aug_type)

    full_dataset = datasets.ImageFolder(root=config.DATA_ROOT, transform=None)

    train_size = int(config.TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 类别分布图
    train_labels = [train_dataset.dataset.samples[i][1] for i in train_dataset.indices]
    train_samples = [(train_dataset.dataset.samples[i][0], train_dataset.dataset.samples[i][1])
                     for i in train_dataset.indices]
    plot_class_distribution(train_samples, config.CLASS_NAMES, config.CLASS_DISTRIBUTION_PATH)

    # 不均衡处理
    if use_weighted_sampler:
        label_counts = Counter(train_labels)
        num_samples = len(train_labels)
        class_weights = {cls: num_samples / count for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"Aug: {aug_type} | Batch: {config.BATCH_SIZE}")
    if use_weighted_sampler:
        min_c, max_c = label_counts.most_common()[-1][1], label_counts.most_common()[0][1]
        print(f"Class imbalance ({min_c}~{max_c}) -> Weighted sampler ON")

    return train_loader, val_loader, full_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, return_preds=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if return_preds:
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if return_preds:
        return running_loss / total, 100.0 * correct / total, all_labels, all_preds
    return running_loss / total, 100.0 * correct / total


def plot_learning_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Curve saved: {save_path}")


def run_single_experiment(exp_config, model_save_path, curve_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = get_data_loaders(
        aug_type=exp_config['augmentation'],
        use_weighted_sampler=config.USE_WEIGHTED_SAMPLER
    )

    model = create_model(num_classes=config.NUM_CLASSES, pretrained=exp_config['use_pretrained']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=exp_config['lr'], weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"{'轮次':<8}{'训练损失':<12}{'训练准确率':<14}{'验证损失':<12}{'验证准确率':<14}{'备注':<6}")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'classes': classes,
                'exp_name': exp_config['name']
            }, model_save_path)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"{epoch:<8}{train_loss:<12.4f}{train_acc:<11.2f}%"
                  f"{val_loss:<12.4f}{val_acc:<11.2f}%{'  ★' if is_best else ''}")

    print(f"\n>>> Best Val Acc: {best_val_acc:.2f}%")

    # 最终评估
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    _, _, all_labels, all_preds = validate(model, val_loader, criterion, device, return_preds=True)
    metrics = compute_metrics(all_labels, all_preds, config.CLASS_NAMES,
                              os.path.join(config.OUTPUT_DIR, f"report_{exp_config['name']}.txt"))
    return {**metrics, 'name': exp_config['name'], 'best_val_acc': best_val_acc,
            'history': history, 'all_labels': all_labels, 'all_preds': all_preds}


def main():
    print(f"  Garbage Classification Training")
    #print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" \nModel: MobileNetV2 | Dataset: 6 classes")

    set_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    experiment_results = []
    for i, exp_config in enumerate(config.EXPERIMENTS):
        print(f"  [{i+1}/{len(config.EXPERIMENTS)}] {exp_config['name']}")
        print(f"  lr={exp_config['lr']}, aug={exp_config['augmentation']}")

        model_path = os.path.join(config.OUTPUT_DIR, f"model_{exp_config['name']}.pth")
        curve_path = os.path.join(config.OUTPUT_DIR, f"curve_{exp_config['name']}.png")

        result = run_single_experiment(exp_config, model_path, curve_path)
        experiment_results.append(result)
        plot_learning_curves(result['history'], curve_path)
        plot_confusion_matrix(result['all_labels'], result['all_preds'],
                              config.CLASS_NAMES,
                              os.path.join(config.OUTPUT_DIR, f"cm_{exp_config['name']}.png"))

    # 汇总
    print(f" \n对比实验汇总")
    print(f"{'实验名称':<25}{'准确率':<10}{'F1分数':<10}{'精确率':<10}")
    for r in experiment_results:
        print(f"{r['name']:<25}{r['accuracy']*100:<10.2f}%{r['f1_score']*100:<10.2f}%{r['precision']*100:<10.2f}%")

    plot_comparison_results(experiment_results, os.path.join(config.OUTPUT_DIR, "comparison_results.png"))

    best = max(experiment_results, key=lambda x: x['accuracy'])
    shutil.copy(os.path.join(config.OUTPUT_DIR, f"model_{best['name']}.pth"), config.MODEL_SAVE_PATH)
    print(f"\nBest: {best['name']} ({best['accuracy']*100:.2f}%)")
    print(f"Model saved: {config.MODEL_SAVE_PATH}")
    print(f"\nAll outputs in: {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()