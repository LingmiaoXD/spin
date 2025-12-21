import torch
from torch import nn
from typing import Optional, Union


class DTStateFilter(nn.Module):
    """
    双时间尺度状态滤波器（Dual-Timescale State Filter, DTSF）
    使用隐状态 z(t) 平滑建模拥堵程度，同时自适应基础速度阈值。

    z(t) = gamma * z(t-1) + (1 - gamma) * sigmoid((v_th - v_obs) / delta)
    若 v_obs 缺失：z(t) = gamma * z(t-1)
    """

    def __init__(
        self,
        gamma: float = 0.9,
        delta: float = 5.0,
        vth_ratio: float = 0.8,
        v_base_init: float = 60.0,
        initial_z: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.vth_ratio = nn.Parameter(torch.tensor(vth_ratio, dtype=torch.float32))
        self.register_buffer("z", torch.tensor([initial_z], dtype=torch.float32, device=device))
        self.register_buffer("v_base", torch.tensor(v_base_init, dtype=torch.float32, device=device))

    def forward(self, v_obs: Optional[float]):
        # 更新基础速度（慢尺度）
        if v_obs is not None:
            v_tensor = torch.tensor(float(v_obs), dtype=torch.float32, device=self.v_base.device)
            self.v_base.mul_(0.995).add_(0.005 * v_tensor)

        v_th = self.vth_ratio * self.v_base

        if v_obs is not None:
            logits = (v_th - v_tensor) / self.delta
            z_new = self.gamma * self.z + (1 - self.gamma) * torch.sigmoid(logits)
        else:
            # 无观测：仅衰减
            z_new = self.gamma * self.z

        self.z.copy_(z_new)
        # 返回克隆以避免下游意外地原地修改
        return self.z.clone()







