import torch
import torch.nn as nn
import torch.nn.functional as F
from grl import WarmStartGradientReverseLayer


class NuclearWassersteinDiscrepancy(nn.Module):
    """自定义的损失函数，用于实现核范数Wasserstein差异的功能，继承自Module

    Args:
        nn (_type_): _description_
    """
    def __init__(self, classifier: nn.Module):
        """初始化损失函数的属性

        Args:
            classifier (nn.Module): 一个神经网络模型，用于预测输入特征的类别概率
        """
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier

    
    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """用于计算NWD的值

        Args:
            y_s (torch.Tensor): 表示源域的类别概率
            y_t (torch.Tensor): 表示目标域的类别概率

        Returns:
            torch.Tensor: NWD的值
        """
        # 对源域和目标域的类别概率进行softmax归一化,得到预测的概率分布
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        # 计算NWD的值，用目标域的核范数减去源域的核范数，再除以目标域的样本数
        loss = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / y_t.shape[0]
        return loss  # NWD的值
    

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """定义损失函数的前向计算

        Args:
            f (torch.Tensor): 表示模型的中间特征

        Returns:
            torch.Tensor: 损失函数的值
        """
        f_grl = self.grl(f)  # 将特征作为GRL的输入，得到反转后的特征
        y = self.classifier(f_grl)  # 将反转后的特征作为分类器的输入，得到类别概率
        y_s, y_t = y.chunk(2, dim=0)  # 将类别概率按照源域和目标域分开，得到两个张量

        loss = self.n_discrepancy(y_s, y_t)
        return loss