# config.py
"""
配置参数
"""

# 训练配置
TRAIN_CONFIG = {
    # 数据配置
    'data_dir': './CVC-ClinicDB_datasets',
    'image_size': 224,
    'train_ratio': 0.8,
    'batch_size': 8,
    'num_workers': 4,

    # 模型配置
    'dim': 128,
    'num_classes': 1,
    'in_channels': 3,
    'use_moe': True,
    'num_experts': 4,
    'slots_per_expert': 1,
    'moe_stages': [2, 3],  # 在第2和第3阶段使用MoE
    'moe_layers_each_stage': [[2, 4, 6], [2, 4, 6, 8, 10, 12, 14, 16]],  # 各阶段使用MoE的层
    'router_temp': 0.1,
    'aux_loss_weight': 0.08,
    'z_loss_weight': 0.001,
    'moe_loss_weight': 0.1,

    # 训练配置
    'epochs': 100,
    'learning_rate': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'aux_weight1': 0.4,
    'aux_weight2': 0.4,

    # 其他配置
    'seed': 42,
    'save_interval': 10,
    'output_dir': './checkpoints'
}

# 推理配置
INFERENCE_CONFIG = {
    'model_path': './checkpoints/best_model.pth',
    'threshold': 0.5,
    'device': 'cuda'  # 或 'cpu'
}