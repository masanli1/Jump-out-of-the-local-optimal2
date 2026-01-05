# export_model.py
import torch
import sys

sys.path.append('.')
from 跳槽最优 import UNetWithTrueComplement


def export_model_to_onnx():
    """将训练好的模型导出为ONNX格式"""

    # 模型配置（必须和训练时一致）
    config = {
        'dim': 128,
        'num_classes': 1,
        'in_channels': 3,
        'use_moe': True,
        'num_experts': 4,
        'slots_per_expert': 1,
        'moe_stages': [2, 3],
        'moe_layers_each_stage': [[2, 4, 6], [2, 4, 6, 8, 10, 12, 14, 16]],
        'router_temp': 0.1,
        'aux_loss_weight': 0.08,
        'z_loss_weight': 0.001,
        'moe_loss_weight': 0.1
    }

    # 创建模型实例
    model = UNetWithTrueComplement(**config)

    # 加载训练好的权重
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建示例输入
    batch_size = 1
    channels = 3
    height = 224
    width = 224
    dummy_input = torch.randn(batch_size, channels, height, width)

    # 导出为ONNX
    onnx_path = './checkpoints/medical_segmentation_model.onnx'

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )

    print(f"✅ 模型已成功导出到: {onnx_path}")
    print(f"   输入尺寸: {batch_size}x{channels}x{height}x{width}")
    print(f"   输出尺寸: 分割概率图")

    # 验证导出的模型
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX模型验证通过")


if __name__ == "__main__":
    export_model_to_onnx()