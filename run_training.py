# run_training.py
"""
启动训练脚本
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='训练医学图像分割模型')
    parser.add_argument('--config', type=str, default='config.py', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='./CVC-ClinicDB_datasets', help='数据目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--use_moe', action='store_true', help='使用MoE')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='输出目录')

    args = parser.parse_args()

    # 动态导入配置
    if os.path.exists(args.config):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.TRAIN_CONFIG
    else:
        # 使用默认配置
        config = {
            'data_dir': args.data_dir,
            'image_size': 224,
            'train_ratio': 0.8,
            'batch_size': args.batch_size,
            'num_workers': 4,
            'dim': 128,
            'num_classes': 1,
            'in_channels': 3,
            'use_moe': args.use_moe,
            'num_experts': 4,
            'slots_per_expert': 1,
            'moe_stages': [2, 3],
            'moe_layers_each_stage': [[2, 4, 6], [2, 4, 6, 8, 10, 12, 14, 16]],
            'router_temp': 0.1,
            'aux_loss_weight': 0.08,
            'z_loss_weight': 0.001,
            'moe_loss_weight': 0.1,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'min_lr': 1e-6,
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'aux_weight1': 0.4,
            'aux_weight2': 0.4,
            'seed': 42,
            'save_interval': 10,
            'output_dir': args.output_dir
        }

    # 导入并运行训练
    sys.path.append('.')
    from train_segmentation import train_model

    # 训练模型
    print(f"开始训练，配置如下:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    model, history = train_model(config)

    print("训练完成！")


if __name__ == "__main__":
    main()