import os

# ==================== 路径配置 ====================
# 数据集根目录（6个类别子文件夹的父目录）
DATA_ROOT = os.path.join("data", "garbage_classification")

# 输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型保存路径
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# 学习曲线保存路径
LEARNING_CURVE_PATH = os.path.join(OUTPUT_DIR, "learning_curves.png")

# 混淆矩阵保存路径
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

# 类别分布图保存路径
CLASS_DISTRIBUTION_PATH = os.path.join(OUTPUT_DIR, "class_distribution.png")

# 分类报告保存路径
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")

# ==================== 数据集配置 ====================
# 类别名称（按字母顺序，与文件夹名对应）
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# 类别中文名（用于推理显示）
CLASS_NAMES_CN = ["纸板", "玻璃", "金属", "纸张", "塑料", "其他垃圾"]

# 类别数量
NUM_CLASSES = 6

# 训练集/验证集划分比例（8:2）
TRAIN_RATIO = 0.8

# 输入图像尺寸
IMAGE_SIZE = 224

# 是否使用加权采样（处理类别不均衡）
USE_WEIGHTED_SAMPLER = True

# ==================== 训练超参数 ====================
BATCH_SIZE = 16        # CPU训练
NUM_EPOCHS = 25        # 训练轮数
LEARNING_RATE = 0.001  # 初始学习率
WEIGHT_DECAY = 1e-4    # L2正则化

# ==================== 对比实验配置 ====================
# 实验1：默认配置
# 实验2：不同数据增强策略（更弱的增强）
# 实验3：不同学习率

EXPERIMENTS = [
    {
        "name": "实验1_默认配置",
        "lr": 0.001,
        "augmentation": "strong",  # 强增强
        "use_pretrained": True,
    },
    {
        "name": "实验2_无增强",
        "lr": 0.001,
        "augmentation": "weak",  # 弱增强（仅resize+归一化）
        "use_pretrained": True,
    },
    {
        "name": "实验3_低学习率",
        "lr": 0.0001,
        "augmentation": "strong",
        "use_pretrained": True,
    },
]

# ==================== 其他配置 ====================
RANDOM_SEED = 42
NUM_WORKERS = 0  # Windows下设为0