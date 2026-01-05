# inference.py
"""
推理脚本
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from train_segmentation import get_transforms, calculate_metrics


def load_model(model_path, config, device='cuda'):
    """加载训练好的模型"""
    from 跳槽最优 import UNetWithTrueComplement

    model = UNetWithTrueComplement(
        dim=config['dim'],
        n_class=config['num_classes'],
        in_ch=config['in_channels'],
        use_moe=config.get('use_moe', False),
        num_experts=config.get('num_experts', 4),
        slots_per_expert=config.get('slots_per_expert', 1),
        moe_stages=config.get('moe_stages', None),
        moe_layers_each_stage=config.get('moe_layers_each_stage', None),
        router_temp=config.get('router_temp', 0.1),
        aux_loss_weight=config.get('aux_loss_weight', 0.08),
        z_loss_weight=config.get('z_loss_weight', 0.001),
        moe_loss_weight=config.get('moe_loss_weight', 0.1)
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def predict_single_image(model, image_path, transform, device='cuda', threshold=0.5):
    """预测单张图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # 预处理
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        if len(outputs) == 4:
            pred, _, _, _ = outputs
        else:
            pred, _, _ = outputs

        pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]
        pred_mask = (pred_prob > threshold).astype(np.float32)

    # 调整回原始尺寸
    pred_prob_resized = np.array(Image.fromarray(pred_prob).resize(original_size, Image.BILINEAR))
    pred_mask_resized = np.array(Image.fromarray(pred_mask).resize(original_size, Image.NEAREST))

    return {
        'original': image_np,
        'probability': pred_prob_resized,
        'mask': pred_mask_resized,
        'original_size': original_size
    }


def visualize_prediction(results, save_path=None):
    """可视化预测结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(results['original'])
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    axes[1].imshow(results['probability'], cmap='hot')
    axes[1].set_title('预测概率图')
    axes[1].axis('off')

    axes[2].imshow(results['mask'], cmap='gray')
    axes[2].set_title('预测掩码')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def main():
    # 配置
    model_path = './checkpoints/best_model.pth'
    image_path = './test_image.png'  # 替换为你的测试图像路径
    output_dir = './predictions'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载配置
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print(f"加载模型: {model_path}")
    model = load_model(model_path, config, device)

    # 预处理
    transform = get_transforms(config['image_size'], 'val')

    # 预测
    print(f"处理图像: {image_path}")
    results = predict_single_image(model, image_path, transform, device)

    # 可视化
    save_path = os.path.join(output_dir, 'prediction.png')
    visualize_prediction(results, save_path)

    # 保存结果
    prob_img = Image.fromarray((results['probability'] * 255).astype(np.uint8))
    mask_img = Image.fromarray((results['mask'] * 255).astype(np.uint8))

    prob_img.save(os.path.join(output_dir, 'probability.png'))
    mask_img.save(os.path.join(output_dir, 'mask.png'))

    print(f"结果保存到: {output_dir}")


if __name__ == "__main__":
    main()