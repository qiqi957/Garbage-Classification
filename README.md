# 🗑️ 垃圾分类识别系统

基于MobileNetV2的6类垃圾分类模型，PyTorch实现。验证集准确率**94.27%**。

## 📊 模型性能

| 指标 | 准确率 |
|------|--------|
| Accuracy | **94.27%** |
| Precision | 94.40% |
| Recall | 94.27% |
| F1-Score | 94.30% |

### 各类别准确率

| 类别 | 纸板 | 玻璃 | 金属 | 纸张 | 塑料 | 其他 |
|------|------|------|------|------|------|------|
| 准确率 | 96.1% | 93.3% | 91.9% | 95.7% | 95.5% | 90.9% |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.x
- torchvision
- opencv-python
- pillow
- matplotlib
- scikit-learn
- numpy

### 安装依赖

```bash
pip install torch torchvision opencv-python pillow matplotlib scikit-learn numpy

数据集准备
从Kaggle下载 Garbage Classification Dataset

数据集目录结构：
    data/
    ├── cardboard/    # 纸板
    ├── glass/        # 玻璃
    ├── metal/        # 金属
    ├── paper/        # 纸张
    ├── plastic/      # 塑料
    └── trash/        # 其他垃圾

📁 文件说明
    文件	用途
    config.py	配置文件（路径、超参数、实验设置）
    sign_classifier.py	模型定义（MobileNetV2）
    train.py	训练脚本（自动运行3组对比实验）
    predict.py	推理脚本（单图/批量/摄像头）
    utils.py	工具函数（增强、指标、绘图）

🏃 训练模型
bash
python train.py
训练完成后在 output/ 目录生成：

    文件	说明
    best_model.pth	最佳模型权重（准确率94.27%）
    class_distribution.png	类别分布图
    curve_*.png	学习曲线
    cm_*.png	混淆矩阵
    comparison_results.png	对比实验图
    report_*.txt	分类报告
🔍 推理测试
bash
python predict.py
三种模式可选：

    模式	说明
    单张图片推理	输入图片路径，输出类别+置信度
    批量测试	自动每类选1张生成结果截图（保存在 output/test_results/）
    摄像头实时推理	实时识别，按 Q 退出
📈 对照实验结果
    实验	学习率	数据增强	准确率
    实验1（默认）	0.001	✅	90.51%
    实验2（无增强）	0.001	❌	89.72%
    实验3（最佳）	0.0001	✅	94.27%
    💡 小学习率（0.0001）+ 数据增强 = 最佳效果

🛠️ 技术方案
    组件	说明
    骨干网络	MobileNetV2（ImageNet预训练）
    输入尺寸	224×224×3
    优化器	Adam
    损失函数	CrossEntropyLoss
    学习率调度	ReduceLROnPlateau
    Batch Size	16
    Epochs	25
数据增强策略
    随机水平翻转（p=0.5）

    随机旋转（±15°）

    颜色抖动（±20%）

    ImageNet标准化
