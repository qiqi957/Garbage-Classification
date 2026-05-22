"""
推理脚本 - 支持单张图片推理、批量测试、摄像头实时推理
"""
import os
import sys
import random
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from torchvision import transforms

import config
from sign_classifier import create_model


# 推理时的图像预处理
PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_trained_model(model_path=config.MODEL_SAVE_PATH):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        print("请先运行 train.py 训练模型")
        sys.exit(1)

    model = create_model(num_classes=config.NUM_CLASSES, pretrained=False)
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载成功 (验证准确率: {checkpoint.get('val_acc', 'N/A')}%)")
    return model, device


def predict_single_image(model, image_path, device, return_tensor=False):
    """单张图片推理"""
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在 - {image_path}")
        return None

    image = Image.open(image_path).convert("RGB")
    input_tensor = PREDICT_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)

    class_idx = predicted.item()
    confidence = confidence.item()
    class_name = config.CLASS_NAMES[class_idx]
    class_name_cn = config.CLASS_NAMES_CN[class_idx]

    if return_tensor:
        return class_idx, confidence, class_name_cn, input_tensor

    return class_idx, confidence, class_name_cn


def draw_result_on_image(image, class_name_cn, confidence, color=(0, 255, 0)):
    """在图片上绘制识别结果（使用PIL，支持中文）"""
    from PIL import ImageDraw, ImageFont

    # 确保image是PIL格式
    if isinstance(image, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        img = image.copy()

    draw = ImageDraw.Draw(img)

    # 尝试加载中文字体
    font = None
    font_paths = [
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 32)
                break
            except:
                continue

    if font is None:
        font = ImageFont.load_default()

    # 绘制半透明背景条
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, 0), (img.width, 70)], fill=(0, 0, 0, 128))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    # 绘制文字
    text = f"{class_name_cn} ({confidence:.1%})"
    # 黑色边框
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((15 + dx, 18 + dy), text, font=font, fill=(0, 0, 0))
    # 白色文字
    draw.text((15, 18), text, font=font, fill=(255, 255, 255))

    # 转换回OpenCV格式以便显示
    result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return result

def test_random_images(model, device, num_images=5):
    """随机选择不同类别的图片进行测试"""
    print("\n" + "=" * 50)
    print(f"  随机选取 {num_images} 张不同类别图片进行测试")
    print("=" * 50)

    # 从每个类别随机选一张
    selected_images = []
    used_classes = set()

    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.DATA_ROOT, class_name)
        if not os.path.exists(class_dir):
            continue

        images = glob(os.path.join(class_dir, "*.jpg")) + \
                 glob(os.path.join(class_dir, "*.png")) + \
                 glob(os.path.join(class_dir, "*.jpeg"))

        if images:
            selected = random.choice(images)
            selected_images.append((selected, class_name))
            used_classes.add(class_name)

    # 如果某些类别没有图片，补充其他类别
    if len(selected_images) < num_images:
        all_images = []
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(config.DATA_ROOT, class_name)
            if os.path.exists(class_dir):
                all_images.extend(glob(os.path.join(class_dir, "*.jpg")))
                all_images.extend(glob(os.path.join(class_dir, "*.png")))

        remaining = random.sample(all_images,
                                  min(num_images - len(selected_images),
                                      len(all_images)))
        for img_path in remaining:
            class_name = os.path.basename(os.path.dirname(img_path))
            selected_images.append((img_path, class_name))

    # 推理并显示
    output_dir = os.path.join(config.OUTPUT_DIR, "test_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n共测试 {len(selected_images)} 张图片：\n")
    print(f"{'No':<6}{'Filename':<25}{'True':<12}{'Pred':<12}{'Conf':<12}{'Res':<6}")
    print("-" * 75)

    for i, (img_path, true_class) in enumerate(selected_images, 1):
        result = predict_single_image(model, img_path, device)
        if result is None:
            continue

        class_idx, confidence, class_name_cn = result
        true_class_cn = config.CLASS_NAMES_CN[
            config.CLASS_NAMES.index(true_class)
        ]
        correct = "✓" if true_class_cn == class_name_cn else "✗"

        print(f"{i:<6}{os.path.basename(img_path):<25}{true_class_cn:<12}"
              f"{class_name_cn:<12}{confidence:<12.1%}{correct:<6}")

        # 保存标注后的图片
        image = Image.open(img_path).convert("RGB")
        result_img = draw_result_on_image(
            image, class_name_cn, confidence,
            color=(0, 255, 0) if correct == "✓" else (0, 0, 255)
        )
        save_name = f"test_{i}_{true_class}_{class_name_cn}_{confidence:.0%}.jpg"
        cv2.imwrite(os.path.join(output_dir, save_name), result_img)

    print("-" * 75)
    print(f"\n标注图片已保存至: {output_dir}")
    print("可将这些图片截图放入实验报告")


def predict_from_camera(model, device):
    """摄像头实时推理"""
    from PIL import ImageDraw, ImageFont

    print("\n" + "=" * 40)
    print("  摄像头实时推理模式")
    print("  按 Q 键退出")
    print("=" * 40 + "\n")

    # 加载中文字体
    font = None
    font_paths = [
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 28)
                break
            except:
                continue

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头，请检查摄像头连接")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        input_tensor = PREDICT_TRANSFORM(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)

        class_idx = predicted.item()
        confidence = confidence.item()

        if confidence < 0.5:
            result_text = "Uncertain, aim at trash"
            color = (0, 0, 255)
        else:
            class_name_cn = config.CLASS_NAMES_CN[class_idx]
            result_text = f"{class_name_cn} ({confidence:.1%})"
            color = (0, 255, 0)

        # 转换为PIL绘制中文
        pil_frame = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_frame)

        # 半透明背景
        overlay = Image.new('RGBA', pil_frame.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(0, 0), (pil_frame.width, 55)], fill=(0, 0, 0, 160))
        pil_frame = Image.alpha_composite(pil_frame.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_frame)

        # 绘制文字
        if font:
            draw.text((15, 12), result_text, font=font, fill=(255, 255, 255))
        else:
            draw.text((15, 12), result_text, fill=(255, 255, 255))

        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

        cv2.imshow("垃圾分类实时识别 - 按Q退出", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("摄像头推理已退出")

def main():
    """主函数"""
    print("=" * 50)
    print("  垃圾分类模型推理系统")
    print("=" * 50)

    model, device = load_trained_model()

    print("\n请选择推理模式:")
    print("  1. 单张图片推理")
    print("  2. 批量测试（随机6张不同类别，生成报告用截图）")
    print("  3. 摄像头实时推理")
    choice = input("\n请输入选项 (1/2/3，默认2): ").strip() or "2"

    if choice == "1":
        image_path = input("请输入图片路径: ").strip()
        class_idx, confidence, class_name_cn = predict_single_image(
            model, image_path, device
        )
        if class_idx is not None:
            print(f"\n预测结果: {class_name_cn} (置信度: {confidence:.2%})")

            image = Image.open(image_path).convert("RGB")
            result_img = draw_result_on_image(image, class_name_cn, confidence)
            cv2.imshow("预测结果", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif choice == "2":
        test_random_images(model, device, num_images=6)

    elif choice == "3":
        predict_from_camera(model, device)

    else:
        print("无效选项")


if __name__ == "__main__":
    main()