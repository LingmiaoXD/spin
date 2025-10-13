from typing import Type, Mapping, Callable, Optional, Union, List

import torch
import pytorch_lightning as pl
from torchmetrics import Metric
from tsl.imputers import Imputer
from tsl.predictors import Predictor

from ..utils import k_hop_subgraph_sampler


class SPINImputer(pl.LightningModule):

    def __init__(self,
                 model_class: Type,
                 model_kwargs: Mapping,
                 optim_class: Type,
                 optim_kwargs: Mapping,
                 loss_fn: Callable,
                 scale_target: bool = True,
                 whiten_prob: Union[float, List[float]] = 0.2,
                 n_roots_subgraph: Optional[int] = None,
                 n_hops: int = 2,
                 max_edges_subgraph: Optional[int] = 1000,
                 cut_edges_uniformly: bool = False,
                 prediction_loss_weight: float = 1.0,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 scheduler_class: Optional = None,
                 scheduler_kwargs: Optional[Mapping] = None):
        super(SPINImputer, self).__init__()
        
        # 保存参数
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.loss_fn = loss_fn
        self.scale_target = scale_target
        self.whiten_prob = whiten_prob
        self.prediction_loss_weight = prediction_loss_weight
        self.metrics = metrics or {}
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}
        
        # 子图采样参数
        self.n_roots = n_roots_subgraph
        self.n_hops = n_hops
        self.max_edges_subgraph = max_edges_subgraph
        self.cut_edges_uniformly = cut_edges_uniformly
        
        # 创建模型
        self.model = model_class(**model_kwargs)
        
        # 创建指标
        self.train_metrics = {name: metric.clone() for name, metric in self.metrics.items()}
        self.val_metrics = {name: metric.clone() for name, metric in self.metrics.items()}
        self.test_metrics = {name: metric.clone() for name, metric in self.metrics.items()}
        
        # 将metrics移动到GPU（如果可用）
        if torch.cuda.is_available():
            for metric in self.train_metrics.values():
                metric.to('cuda')
            for metric in self.val_metrics.values():
                metric.to('cuda')
            for metric in self.test_metrics.values():
                metric.to('cuda')

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                }
            }
        return optimizer

    def forward(self, x, u=None, mask=None, edge_index=None, edge_weight=None):
        """前向传播方法，PyTorch Lightning 的 predict 需要"""
        return self.model(x, u, mask, edge_index, edge_weight)

    def predict_batch(self, batch, preprocess=True, postprocess=True):
        """预测批次数据"""
        # 尝试不同的方式访问batch数据
        try:
            # 方式1：直接访问batch属性
            x = batch.x
            u = batch.u if hasattr(batch, 'u') else None
            mask = batch.mask
            edge_index = batch.edge_index
            edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None
        except AttributeError:
            try:
                # 方式2：通过batch.input访问
                x = batch.input.x
                u = batch.input.u if hasattr(batch.input, 'u') else None
                mask = batch.input.mask
                edge_index = batch.input.edge_index
                edge_weight = batch.input.edge_weight if hasattr(batch.input, 'edge_weight') else None
            except AttributeError:
                # 方式3：通过batch.data访问
                x = batch.data.x
                u = batch.data.u if hasattr(batch.data, 'u') else None
                mask = batch.data.mask
                edge_index = batch.data.edge_index
                edge_weight = batch.data.edge_weight if hasattr(batch.data, 'edge_weight') else None
        
        # 调用模型
        output = self.model(x, u, mask, edge_index, edge_weight)
        
        if isinstance(output, (list, tuple)):
            return output
        else:
            return output, []

    def trim_warm_up(self, y_hat, y, mask):
        """修剪预热期"""
        return y_hat, y, mask

    def log_metrics(self, metrics_dict, batch_size):
        """记录指标"""
        for name, metric in metrics_dict.items():
            # 计算指标值
            metric_value = metric.compute()
            # 确保指标值在CPU上用于logging
            if hasattr(metric_value, 'cpu'):
                metric_value = metric_value.cpu()
            stage = 'train' if self.training else 'val'
            self.log(f'{stage}_{name}', metric_value, batch_size=batch_size)
            # 重置指标
            metric.reset()

    def log_loss(self, stage, loss, batch_size):
        """记录损失"""
        self.log(f'{stage}_loss', loss, batch_size=batch_size)

    def get_device(self):
        """获取模型所在的设备"""
        return next(self.model.parameters()).device

    def ensure_tensor_on_device(self, tensor, device=None):
        """确保张量在指定设备上"""
        if device is None:
            device = self.get_device()
        if hasattr(tensor, 'to'):
            return tensor.to(device)
        return tensor

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.training and self.n_roots is not None:
            batch = k_hop_subgraph_sampler(batch, self.n_hops, self.n_roots,
                                           max_edges=self.max_edges_subgraph,
                                           cut_edges_uniformly=self.cut_edges_uniformly)
        return batch

    def shared_step(self, batch, mask):
        """简化的shared_step方法"""
        # 获取设备信息
        device = next(self.model.parameters()).device
        
        y = batch.y.clone().detach().to(device)
        mask = mask.clone().detach().to(device)
        
        y_hat_loss = self.predict_batch(batch, preprocess=False, postprocess=False)

        if isinstance(y_hat_loss, (list, tuple)):
            imputation, predictions = y_hat_loss
            y_hat = imputation.to(device)
            predictions = [pred.to(device) for pred in predictions]
        else:
            imputation = y_hat_loss.to(device)
            predictions = []
            y_hat = imputation

        # 计算损失
        if self.training:
            # 检查batch是否有original_mask属性
            if hasattr(batch, 'original_mask'):
                injected_missing = (batch.original_mask - batch.mask).clone().detach().to(device)
                loss = self.loss_fn(imputation, y, injected_missing)
            else:
                loss = self.loss_fn(imputation, y, mask)
        else:
            loss = self.loss_fn(imputation, y, mask)

        # 添加预测损失
        for pred in predictions:
            pred_loss = self.loss_fn(pred, y, mask)
            loss += self.prediction_loss_weight * pred_loss / 3

        return y_hat.detach(), y, loss

    def training_step(self, batch, batch_idx):
        # 获取设备信息
        device = self.get_device()
        
        # 在训练时，使用mask作为缺失值指示器
        # mask表示哪些值是可观测的（1表示可观测，0表示缺失）
        # 我们在这些可观测的值上计算损失
        if hasattr(batch, 'mask'):
            injected_missing = batch.mask.clone().detach().to(device)
        else:
            # 如果没有mask，尝试使用eval_mask
            injected_missing = batch.eval_mask.clone().detach().to(device)
        
        if hasattr(batch, 'target_nodes'):
            injected_missing = injected_missing[..., batch.target_nodes, :]
        
        y_hat, y, loss = self.shared_step(batch, mask=injected_missing)

        # 更新指标 - 确保张量在正确设备上
        for name, metric in self.train_metrics.items():
            y_hat_device = self.ensure_tensor_on_device(y_hat)
            y_device = self.ensure_tensor_on_device(y)
            metric.update(y_hat_device, y_device)

        # Logging
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        
        if hasattr(batch, 'target_nodes'):
            torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        # 获取设备信息
        device = next(self.model.parameters()).device
        
        # 确保mask不共享内存并移到正确设备
        eval_mask = batch.eval_mask.clone().detach().to(device)
        y_hat, y, val_loss = self.shared_step(batch, eval_mask)

        # 更新指标 - 确保张量在正确设备上
        for name, metric in self.val_metrics.items():
            y_hat_device = self.ensure_tensor_on_device(y_hat)
            y_device = self.ensure_tensor_on_device(y)
            metric.update(y_hat_device, y_device)

        # Logging
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        # 获取设备信息
        device = next(self.model.parameters()).device
        
        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]

        y, eval_mask = batch.y.to(device), batch.eval_mask.to(device)
        y_hat = y_hat.to(device)
        test_loss = self.loss_fn(y_hat, y, eval_mask)

        # 更新指标 - 确保张量在正确设备上
        for name, metric in self.test_metrics.items():
            y_hat_device = self.ensure_tensor_on_device(y_hat)
            y_device = self.ensure_tensor_on_device(y)
            metric.update(y_hat_device, y_device)

        # Logging
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """预测步骤，用于 Trainer.predict()"""
        # 获取设备信息
        device = next(self.model.parameters()).device
        
        # 执行预测
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)
        
        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]
        
        # 获取真实值和mask
        y = batch.y.to(device)
        eval_mask = batch.eval_mask.to(device)
        y_hat = y_hat.to(device)
        
        # 返回预测结果、真实值和mask
        return {
            'y_hat': y_hat,
            'y': y,
            'mask': eval_mask
        }

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser = Predictor.add_argparse_args(parser)
        parser.add_argument('--whiten-prob', type=float, default=0.05)
        parser.add_argument('--prediction-loss-weight', type=float, default=1.0)
        parser.add_argument('--n-roots-subgraph', type=int, default=None)
        parser.add_argument('--n-hops', type=int, default=2)
        parser.add_argument('--max-edges-subgraph', type=int, default=1000)
        parser.add_argument('--cut-edges-uniformly', type=bool, default=False)
        return parser
