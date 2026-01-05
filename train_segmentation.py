# train_segmentation.py - torchvision修复版
import os
import sys
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tqdm

# 添加当前目录到路径
sys.path.append('.')


# ============================================================
# 数据集类 - 使用torchvision替代albumentations
# ============================================================

class CVCClinicDBDataset(Dataset):
    """CVC-ClinicDB数据集加载器"""

    def __init__(self, root_dir, transform=None, split='train', train_ratio=0.8,
                 seed=42, image_size=224):
        """
        Args:
            root_dir: 数据集根目录
            transform: 数据增强变换
            split: 'train', 'val', 'test'
            train_ratio: 训练集比例
            seed: 随机种子
            image_size: 图像尺寸
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_size = image_size

        # 加载PNG数据集
        png_dir = os.path.join(root_dir, 'CVC-ClinicDB_PNG_datasets')
        images_dir = os.path.join(png_dir, 'Original')
        masks_dir = os.path.join(png_dir, 'Ground Truth')

        # 收集所有图像和掩码路径
        self.image_paths = []
        self.mask_paths = []

        # 假设图像和掩码文件名相同
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file)

            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)

        print(f"找到 {len(self.image_paths)} 对图像-掩码")

        # 划分数据集
        indices = list(range(len(self.image_paths)))
        random.seed(seed)
        random.shuffle(indices)

        train_size = int(len(indices) * train_ratio)
        val_size = len(indices) - train_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        if split == 'train':
            self.indices = train_indices
        elif split == 'val':
            self.indices = val_indices
        elif split == 'all':
            self.indices = indices
        else:
            raise ValueError(f"无效的split: {split}")

        print(f"{split}集大小: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        # 加载图像和掩码
        image = Image.open(self.image_paths[actual_idx]).convert('RGB')
        # 确保掩码以单通道模式加载
        mask = Image.open(self.mask_paths[actual_idx]).convert('L')

        # 转换为PIL图像后直接应用torchvision变换
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # 基础转换
            image = transforms.ToTensor()(image)
            # 确保mask是二值图像且为单通道
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()
            # 如果mask有多个通道，只取第一个通道
            if mask.dim() == 3 and mask.shape[0] > 1:
                mask = mask[0:1, :, :]

        return image, mask


class Transform:
    """自定义变换类，同时处理图像和掩码"""

    def __init__(self, image_size=224, split='train'):
        self.image_size = image_size
        self.split = split
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size))
            ])

    def __call__(self, image, mask):
        # 确保掩码是单通道
        if mask.mode != 'L':
            mask = mask.convert('L')

        # 应用相同的空间变换
        seed = torch.randint(0, 2 ** 32, (1,)).item()

        # 对图像应用变换
        torch.manual_seed(seed)
        image = self.transform(image)

        # 对掩码应用相同的变换（不使用颜色jitter）
        if self.split == 'train':
            # 对掩码只应用空间变换
            mask_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15)
            ])
            torch.manual_seed(seed)
            mask = mask_transform(mask)
        else:
            mask = transforms.Resize((self.image_size, self.image_size))(mask)

        # 转换为张量
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        # 归一化图像
        image = self.normalize(image)

        # 确保mask是二值图像且为单通道
        mask = (mask > 0.5).float()
        # 如果mask有多个通道，只取第一个通道
        if mask.dim() == 3 and mask.shape[0] > 1:
            mask = mask[0:1, :, :]

        return image, mask


def get_transforms(image_size=224, split='train'):
    """获取数据增强变换 - 使用torchvision"""
    return Transform(image_size, split)


# ============================================================
# 训练和验证函数
# ============================================================

def train_epoch(model, dataloader, optimizer, device, config, compute_metrics=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux1_loss = 0
    total_aux2_loss = 0
    total_moe_loss = 0

    # 可选：累计训练指标
    if compute_metrics:
        all_iou = []
        all_dice = []
        all_accuracy = []
        all_precision = []
        all_recall = []

    progress_bar = tqdm.tqdm(dataloader, desc='训练')

    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        if len(outputs) == 4:
            pred, aux1, aux2, moe_loss = outputs
            loss, main_loss, aux1_loss, aux2_loss, moe_loss = calculate_total_loss(
                (pred, aux1, aux2, moe_loss), masks, config
            )
        else:
            pred, aux1, aux2 = outputs
            moe_loss = torch.tensor(0.0).to(device)
            loss, main_loss, aux1_loss, aux2_loss, moe_loss = calculate_total_loss(
                (pred, aux1, aux2), masks, config
            )

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux1_loss += aux1_loss.item()
        total_aux2_loss += aux2_loss.item()
        total_moe_loss += moe_loss.item()

        # 可选：计算训练指标
        if compute_metrics:
            metrics = calculate_metrics(pred, masks)
            all_iou.append(metrics['iou'])
            all_dice.append(metrics['dice'])
            all_accuracy.append(metrics['accuracy'])
            all_precision.append(metrics['precision'])
            all_recall.append(metrics['recall'])

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'main': f'{main_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_aux1_loss = total_aux1_loss / len(dataloader)
    avg_aux2_loss = total_aux2_loss / len(dataloader)
    avg_moe_loss = total_moe_loss / len(dataloader)

    result = {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'aux1_loss': avg_aux1_loss,
        'aux2_loss': avg_aux2_loss,
        'moe_loss': avg_moe_loss
    }

    # 可选：添加训练指标
    if compute_metrics:
        result['iou'] = np.mean(all_iou) if all_iou else 0
        result['dice'] = np.mean(all_dice) if all_dice else 0
        result['accuracy'] = np.mean(all_accuracy) if all_accuracy else 0
        result['precision'] = np.mean(all_precision) if all_precision else 0
        result['recall'] = np.mean(all_recall) if all_recall else 0

    return result


def validate(model, dataloader, device, config):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_main_loss = 0
    total_aux1_loss = 0
    total_aux2_loss = 0

    # 用于累计指标
    all_iou = []
    all_dice = []
    all_accuracy = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
        progress_bar = tqdm.tqdm(dataloader, desc='验证')

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            if len(outputs) == 4:
                pred, aux1, aux2, moe_loss = outputs
                loss, main_loss, aux1_loss, aux2_loss, _ = calculate_total_loss(
                    (pred, aux1, aux2, moe_loss), masks, config
                )
            else:
                pred, aux1, aux2 = outputs
                loss, main_loss, aux1_loss, aux2_loss, _ = calculate_total_loss(
                    (pred, aux1, aux2), masks, config
                )

            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_aux1_loss += aux1_loss.item()
            total_aux2_loss += aux2_loss.item()

            # 计算指标
            metrics = calculate_metrics(pred, masks)
            all_iou.append(metrics['iou'])
            all_dice.append(metrics['dice'])
            all_accuracy.append(metrics['accuracy'])
            all_precision.append(metrics['precision'])
            all_recall.append(metrics['recall'])

            # 更新进度条显示当前batch的指标
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{metrics["iou"]:.4f}',
                'dice': f'{metrics["dice"]:.4f}'
            })

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_aux1_loss = total_aux1_loss / len(dataloader)
    avg_aux2_loss = total_aux2_loss / len(dataloader)

    # 计算平均指标
    avg_iou = np.mean(all_iou) if all_iou else 0
    avg_dice = np.mean(all_dice) if all_dice else 0
    avg_accuracy = np.mean(all_accuracy) if all_accuracy else 0
    avg_precision = np.mean(all_precision) if all_precision else 0
    avg_recall = np.mean(all_recall) if all_recall else 0

    return {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'aux1_loss': avg_aux1_loss,
        'aux2_loss': avg_aux2_loss,
        'iou': avg_iou,
        'dice': avg_dice,
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall
    }


def calculate_total_loss(outputs, targets, config):
    """计算总损失"""
    if len(outputs) == 4:
        pred, aux1, aux2, moe_loss = outputs
    else:
        pred, aux1, aux2 = outputs
        moe_loss = torch.tensor(0.0).to(pred.device)

    # 确保targets是4维的 [B, 1, H, W]
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    # 如果targets有多个通道，只取第一个通道
    if targets.shape[1] > 1:  # [B, C, H, W] where C > 1
        targets = targets[:, 0:1, :, :]  # [B, 1, H, W]

    # 主损失
    main_loss = structure_loss(pred, targets)

    # 辅助损失
    aux1_loss = structure_loss(aux1, targets)
    aux2_loss = structure_loss(aux2, targets)

    # 总损失
    total_loss = (main_loss +
                  config['aux_weight1'] * aux1_loss +
                  config['aux_weight2'] * aux2_loss +
                  moe_loss)

    return total_loss, main_loss, aux1_loss, aux2_loss, moe_loss


def structure_loss(pred, mask):
    """结构损失函数"""
    # 确保mask有正确的维度 [B, 1, H, W]
    if mask.dim() == 3:  # [B, H, W]
        mask = mask.unsqueeze(1)  # [B, 1, H, W]

    # 如果mask有多个通道，只取第一个通道
    if mask.shape[1] > 1:  # [B, C, H, W] where C > 1
        mask = mask[:, 0:1, :, :]  # [B, 1, H, W]

    # 确保pred和mask维度匹配
    if pred.shape != mask.shape:
        # 如果pred是 [B, 1, H, W] 而 mask是 [B, H, W]，调整mask
        if pred.dim() == 4 and mask.dim() == 3 and pred.shape[0] == mask.shape[0] and pred.shape[2:] == mask.shape[1:]:
            mask = mask.unsqueeze(1)

    # 再次检查维度
    assert pred.shape == mask.shape, f"Pred shape {pred.shape} != Mask shape {mask.shape}"

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# ============================================================
# 评估指标
# ============================================================

def calculate_metrics(pred, target, threshold=0.5):
    """计算评估指标"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()

    # 确保target是4维的
    if target.dim() == 3:
        target = target.unsqueeze(1)

    # 如果target有多个通道，只取第一个通道
    if target.shape[1] > 1:  # [B, C, H, W] where C > 1
        target = target[:, 0:1, :, :]  # [B, 1, H, W]

    # 确保pred和target维度匹配
    if pred_bin.shape != target.shape:
        # 调整pred_bin的维度
        if pred_bin.dim() == 4 and pred_bin.shape[1] > 1:
            pred_bin = pred_bin[:, 0:1, :, :]
        elif pred_bin.dim() == 3:
            pred_bin = pred_bin.unsqueeze(1)

    # 再次检查维度
    if pred_bin.shape != target.shape:
        # 如果形状不匹配，调整pred_bin的大小
        pred_bin = F.interpolate(pred_bin, size=target.shape[2:], mode='bilinear', align_corners=True)

    # 计算IoU
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = (pred_bin + target).clamp(0, 1).sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)

    # 计算Dice系数
    dice = (2 * intersection + 1e-6) / (pred_bin.sum(dim=(1, 2, 3)) +
                                        target.sum(dim=(1, 2, 3)) + 1e-6)

    # 计算准确率、精确率、召回率
    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    tn = ((1 - pred_bin) * (1 - target)).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))

    accuracy = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)

    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'accuracy': accuracy.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item()
    }


def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for images, masks in tqdm.tqdm(dataloader, desc='评估'):
            images = images.to(device)
            masks = masks.to(device)

            # 确保masks是4维的
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            # 如果masks有多个通道，只取第一个通道
            if masks.shape[1] > 1:  # [B, C, H, W] where C > 1
                masks = masks[:, 0:1, :, :]  # [B, 1, H, W]

            # 前向传播
            outputs = model(images)

            if len(outputs) == 4:
                pred, _, _, _ = outputs
            else:
                pred, _, _ = outputs

            # 计算指标
            metrics = calculate_metrics(pred, masks)
            all_metrics.append(metrics)

    # 平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics


# ============================================================
# 模型管理函数
# ============================================================

def cleanup_old_models(output_dir, keep_files=None):
    """
    清理旧的模型文件，只保留指定的文件

    Args:
        output_dir: 输出目录
        keep_files: 要保留的文件列表
    """
    if keep_files is None:
        keep_files = ['best_model.pth', 'final_model.pth', 'training_history.npy']

    if not os.path.exists(output_dir):
        return

    for file in os.listdir(output_dir):
        # 跳过目录
        if os.path.isdir(os.path.join(output_dir, file)):
            continue

        # 检查是否是需要保留的文件
        should_keep = False
        for pattern in keep_files:
            if pattern in file:
                should_keep = True
                break

        # 如果不是需要保留的文件，并且是模型文件，则删除
        if not should_keep and (file.endswith('.pth') or file.endswith('.pt')):
            try:
                os.remove(os.path.join(output_dir, file))
                print(f"🗑️  清理旧文件: {file}")
            except Exception as e:
                print(f"⚠️  无法清理文件 {file}: {e}")


# ============================================================
# 训练循环（精简保存策略）
# ============================================================

def train_model(config):
    """主训练函数 - 精简保存策略，只保留最佳模型和最终模型"""
    print("开始训练...")
    print(f"配置参数: {config}")

    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据集
    print("加载数据集...")
    train_transform = get_transforms(config['image_size'], 'train')
    val_transform = get_transforms(config['image_size'], 'val')

    train_dataset = CVCClinicDBDataset(
        root_dir=config['data_dir'],
        transform=train_transform,
        split='train',
        train_ratio=config['train_ratio'],
        seed=config['seed'],
        image_size=config['image_size']
    )

    val_dataset = CVCClinicDBDataset(
        root_dir=config['data_dir'],
        transform=val_transform,
        split='val',
        train_ratio=config['train_ratio'],
        seed=config['seed'],
        image_size=config['image_size']
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建模型
    print("创建模型...")
    from 跳槽最优 import UNetWithTrueComplement

    model = UNetWithTrueComplement(
        dim=config['dim'],
        n_class=config['num_classes'],
        in_ch=config['in_channels'],
        use_moe=config['use_moe'],
        num_experts=config['num_experts'],
        slots_per_expert=config['slots_per_expert'],
        moe_stages=config['moe_stages'],
        moe_layers_each_stage=config['moe_layers_each_stage'],
        router_temp=config['router_temp'],
        aux_loss_weight=config['aux_loss_weight'],
        z_loss_weight=config['z_loss_weight'],
        moe_loss_weight=config['moe_loss_weight']
    )

    model = model.to(device)

    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['min_lr']
    )

    # 训练记录 - 添加指标记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_main_loss': [],
        'val_main_loss': [],
        'val_iou': [],  # 新增
        'val_dice': [],  # 新增
        'val_accuracy': [],  # 新增
        'val_precision': [],  # 新增
        'val_recall': [],  # 新增
        'train_iou': [],  # 新增（可选）
        'train_dice': [],  # 新增（可选）
        'learning_rate': []
    }

    best_val_loss = float('inf')
    best_val_iou = 0.0  # 新增
    best_epoch = -1

    # 定义模型文件路径
    best_model_path = os.path.join(config['output_dir'], 'best_model.pth')
    final_model_path = os.path.join(config['output_dir'], 'final_model.pth')

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    # 如果配置了清理旧模型，则清理输出目录
    if config.get('clean_old', False):
        print("清理旧的模型文件...")
        cleanup_old_models(config['output_dir'])

    # 训练循环
    print("开始训练循环...")
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # 每5个epoch计算一次训练集指标，避免减慢训练速度
        compute_train_metrics = (epoch % 5 == 0) or (epoch == config['epochs'] - 1)

        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, compute_train_metrics)

        # 验证
        val_metrics = validate(model, val_loader, device, config)

        # 更新学习率
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_main_loss'].append(train_metrics['main_loss'])
        history['val_main_loss'].append(val_metrics['main_loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['learning_rate'].append(scheduler.get_last_lr()[0])

        # 如果计算了训练指标，也记录下来
        if compute_train_metrics and 'iou' in train_metrics:
            history['train_iou'].append(train_metrics['iou'])
            history['train_dice'].append(train_metrics['dice'])

        # 打印结果 - 包含指标
        print(f"训练损失: {train_metrics['loss']:.4f}, 验证损失: {val_metrics['loss']:.4f}")
        print(f"训练主损失: {train_metrics['main_loss']:.4f}, 验证主损失: {val_metrics['main_loss']:.4f}")

        # 打印验证指标
        print(f"验证指标 - IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"准确率: {val_metrics['accuracy']:.4f}, 精确率: {val_metrics['precision']:.4f}, "
              f"召回率: {val_metrics['recall']:.4f}")

        # 如果计算了训练指标，也打印出来
        if compute_train_metrics and 'iou' in train_metrics:
            print(f"训练指标 - IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}")

        # 保存最佳模型 - 使用IoU作为主要指标
        current_val_iou = val_metrics['iou']
        if current_val_iou > best_val_iou:
            # 如果已有最佳模型，先删除旧模型
            if os.path.exists(best_model_path):
                print(f"🎯 发现更优模型 (IoU从 {best_val_iou:.4f} 提升到 {current_val_iou:.4f})")
                print(f"🗑️  删除旧的最佳模型: {best_model_path}")
                os.remove(best_model_path)

            # 保存新的最佳模型
            best_val_iou = current_val_iou
            best_epoch = epoch

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_iou': best_val_iou,
                'val_loss': val_metrics['loss'],
                'train_loss': train_metrics['loss'],
                'config': config
            }, best_model_path)

            print(f"✅ 保存最佳模型到 {best_model_path}")
            print(f"   Epoch: {epoch + 1}, 验证IoU: {best_val_iou:.4f}, 验证损失: {val_metrics['loss']:.4f}")

        # 如果配置了只保留最佳模型，定期清理中间文件
        if config.get('keep_only_best', False) and (epoch + 1) % 5 == 0:
            cleanup_old_models(config['output_dir'])

    # 保存最终模型（覆盖旧的最终模型）
    if os.path.exists(final_model_path):
        print(f"🗑️  删除旧的最终模型: {final_model_path}")
        os.remove(final_model_path)

    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_iou': val_metrics['iou'],
        'val_loss': val_metrics['loss'],
        'train_loss': train_metrics['loss'],
        'config': config,
        'is_final': True
    }, final_model_path)

    print(f"💾 保存最终模型到 {final_model_path}")
    print(
        f"   Epoch: {config['epochs']}, 最终验证IoU: {val_metrics['iou']:.4f}, 最终验证损失: {val_metrics['loss']:.4f}")

    # 保存训练历史
    history_path = os.path.join(config['output_dir'], 'training_history.npy')
    np.save(history_path, history)
    print(f"📈 保存训练历史到 {history_path}")

    # 如果配置了只保留最佳模型，最终清理一次
    if config.get('keep_only_best', False):
        print("清理文件，只保留最佳模型...")
        cleanup_old_models(config['output_dir'], keep_files=['best_model.pth', 'training_history.npy'])

    # 打印最佳模型信息
    if best_epoch >= 0:
        print(f"\n🎯 训练总结:")
        print(f"  最佳模型: Epoch {best_epoch + 1}, 验证IoU: {best_val_iou:.4f}, 验证损失: {best_val_loss:.4f}")
        print(f"  最佳模型路径: {best_model_path}")
        print(f"  最终模型路径: {final_model_path}")
        print(f"  训练历史路径: {history_path}")
    else:
        print("⚠️  未找到最佳模型，可能是训练过程中验证损失没有改善")

    return model, history


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""

    # 配置参数
    config = {
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
        'moe_stages': [2, 3],
        'moe_layers_each_stage': [[2, 4, 6], [2, 4, 6, 8, 10, 12, 14, 16]],
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

        # 模型保存配置（新增）
        'clean_old': True,  # 清理旧的模型文件
        'keep_only_best': False,  # 是否只保留最佳模型

        # 其他配置
        'seed': 42,
        'save_interval': 10,  # 这个参数仍然保留，用于其他用途
        'output_dir': './checkpoints'
    }

    # 训练模型
    model, history = train_model(config)

    # 可视化训练过程
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练和验证损失')

    plt.subplot(1, 5, 2)
    plt.plot(history['val_iou'], label='验证IoU', color='green')
    if 'train_iou' in history and history['train_iou']:
        plt.plot(history['train_iou'], label='训练IoU', color='lightgreen', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('IoU指标')

    plt.subplot(1, 5, 3)
    plt.plot(history['val_dice'], label='验证Dice', color='blue')
    if 'train_dice' in history and history['train_dice']:
        plt.plot(history['train_dice'], label='训练Dice', color='lightblue', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.title('Dice系数')

    plt.subplot(1, 5, 4)
    plt.plot(history['val_accuracy'], label='准确率', color='red')
    plt.plot(history['val_precision'], label='精确率', color='orange')
    plt.plot(history['val_recall'], label='召回率', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('分数')
    plt.legend()
    plt.title('其他指标')

    plt.subplot(1, 5, 5)
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.title('学习率变化')

    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'training_metrics.png'), dpi=150)
    plt.show()

    print("训练完成！")


if __name__ == "__main__":
    main()