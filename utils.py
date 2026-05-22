"""
工具函数 - 数据可视化、评估指标、类别分布统计
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

import config

# 设置字体（使用英文避免中文字体问题）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


def plot_class_distribution(dataset, class_names, save_path):
    """绘制类别分布柱状图"""
    # 兼容 (path, label) 或 (label,) 格式
    labels = [sample[1] if len(sample) >= 2 else sample[0] for sample in dataset]
    counter = Counter(labels)
    counts = [counter.get(i, 0) for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_names, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1',
                                               '#96CEB4', '#FFEAA7', '#DDA0DD'],
                  edgecolor='black', linewidth=0.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Garbage Classification Dataset Distribution', fontsize=15, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    total = sum(counts)
    max_count = max(counts)
    min_count = min(counts)
    ratio = max_count / min_count if min_count > 0 else float('inf')

    ax.text(0.98, 0.95, f'Total: {total}', transform=ax.transAxes,
            fontsize=12, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if ratio > 2:
        ax.text(0.98, 0.82, f'Imbalanced ({ratio:.1f}:1)', transform=ax.transAxes,
                fontsize=10, ha='right', va='top', color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Class distribution saved: {save_path}")

    return counts, ratio


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted',
           ylabel='True',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved: {save_path}")


def compute_metrics(y_true, y_pred, class_names, save_path=None):
    """计算并输出分类指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f}")

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n\n")
            f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
        print(f"[OK] Report saved: {save_path}")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}


def get_data_augmentation(aug_type="strong"):
    """获取数据增强策略"""
    from torchvision import transforms

    if aug_type == "strong":
        train_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def plot_comparison_results(results, save_path):
    """绘制对比实验结果"""
    if not results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    name_map = {
        '实验1_默认配置': 'Exp1: Default',
        '实验2_无增强': 'Exp2: No Aug',
        '实验3_低学习率': 'Exp3: Low LR'
    }
    names = [name_map.get(r['name'], r['name']) for r in results]
    accs = [r['accuracy'] * 100 for r in results]
    f1s = [r['f1_score'] * 100 for r in results]
    precisions = [r['precision'] * 100 for r in results]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for ax, data, title in zip(axes, [accs, f1s, precisions],
                                ['Accuracy (%)', 'F1-Score (%)', 'Precision (%)']):
        ax.bar(names, data, color=colors[:len(names)], edgecolor='black')
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        for i, v in enumerate(data):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Comparison saved: {save_path}")