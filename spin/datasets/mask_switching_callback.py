"""
Mask切换回调，用于在训练时每个epoch随机选择不同的mask文件
避免固定缺失模式形成硬编码式捷径
"""

import pytorch_lightning as pl
from typing import Optional
import numpy as np


class MaskSwitchingCallback(pl.Callback):
    """
    在每个训练epoch开始时随机切换mask的回调
    
    使用方法：
        from spin.datasets.mask_switching_callback import MaskSwitchingCallback
        
        callback = MaskSwitchingCallback(dataset, torch_dataset)
        trainer = pl.Trainer(callbacks=[callback, ...])
    """
    
    def __init__(self, dataset, torch_dataset, seed: Optional[int] = None):
        """
        初始化回调
        
        Args:
            dataset: LaneTrafficDataset实例
            torch_dataset: ImputationDataset实例
            seed: 随机种子，如果为None则使用epoch编号作为种子
        """
        super().__init__()
        self.dataset = dataset
        self.torch_dataset = torch_dataset
        self.seed = seed
        self.epoch_seed_base = 42  # 基础种子，用于确保可重复性
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """在每个训练epoch开始时调用"""
        # 检查是否有mask_files列表
        if not self.dataset.mask_files:
            return
        
        # 计算当前epoch的种子
        current_epoch = trainer.current_epoch
        if self.seed is not None:
            epoch_seed = self.seed + current_epoch
        else:
            epoch_seed = self.epoch_seed_base + current_epoch
        
        # 切换mask
        success = self.dataset.switch_mask_randomly(seed=epoch_seed)
        
        if success:
            # 更新torch_dataset的mask
            if hasattr(self.torch_dataset, 'set_mask'):
                self.torch_dataset.set_mask(self.dataset.training_mask)
            # 更新exogenous中的eval_mask
            if hasattr(self.torch_dataset, 'update_exogenous'):
                self.torch_dataset.update_exogenous('eval_mask', self.dataset.eval_mask)
            print(f"📊 Epoch {current_epoch}: 已切换到mask文件: {self.dataset.current_mask_file}")

