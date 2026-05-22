"""
模型定义 - MobileNetV2 垃圾分类器
"""
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

class GarbageClassifier(nn.Module):
    """基于MobileNetV2的垃圾分类模型，使用迁移学习"""

    def __init__(self, num_classes=6, pretrained=True):
        super(GarbageClassifier, self).__init__()
        #加载预训练MobileNetV2 = features + 原始classifier
        self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        #获取原始分类头的输入通道数 (1280)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),#防止过拟合
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_model(num_classes=6, pretrained=True):
    """工厂函数：创建模型实例"""
    return GarbageClassifier(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    import torch
    model = create_model()
    print(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")  # 应为 [1, 6]