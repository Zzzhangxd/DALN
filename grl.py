from typing import Optional, Any, Tuple
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch


class GradientReverseFunction(Function):
    """用于实现梯度反转层GRL的功能
    前向传播的时候不改变输入，但在反向传播时反转梯度的符号
    Args:
        Function (_type_): _description_
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        """_summary_

        Args:
            ctx (Any): 上下文对象，用于保存一些信息，方便在反向计算时使用
            input (torch.Tensor): 张量，表示GRL的输入
            coeff (Optional[float], optional): 浮点数，表示GRL的系数lambda，默认为1

        Returns:
            torch.Tensor: 表示GRL的输出
        """
        ctx.coeff = coeff  # 将系数lambda保存到ctx中
        output = input * 1.0  # 不改变输入
        return output
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """定义反向计算的逻辑，即GRL的梯度

        Args:
            ctx (Any): 上下文对象，用于获取前向计算时保存的信息
            grad_output (torch.Tensor): 张量，表示GRL输出的梯度

        Returns:
            Tuple[torch.Tensor, Any]: 返回一个元组，表示GRL输入的梯度和其他参数的梯度
        """
        # neg()的作用是将grad_output的进行取负操作，返回一个新的张量，其元素是原张量的负数
        return grad_output.neg() * ctx.coeff, None  # 将输出的梯度取负，再乘以系数，得到输入的梯度，其他参数的梯度为None
    

class GradientReverseLayer(nn.Module):
    """这个类是一个自定义层，用于封装GRL的功能，继承自nn.Module

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(GradientReverseLayer, self).__init__()  # 调用父类的初始化方法

    
    def forward(self, *input):
        """定义层的前向计算的逻辑，层的输入和输出
        """
        return GradientReverseFunction.apply(*input)  # apply继承自Function类， 将张量作为输入，传递给自定义函数forward方法得到输出
    

class WarmStartGradientReverseLayer(nn.Module):
    """用于实现带有温启动的GRL的功能，继承自Module,这里温启动的作用是使得GRL的lambda系数不是固定的，而是根据迭代次数动态变化，从而提高训练效果

    Args:
        nn (_type_): _description_
    """
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        """初始化层的属性

        Args:
            alpha (Optional[float], optional): 表示温启动的调节因子. Defaults to 1.0.
            lo (Optional[float], optional): 浮点数，表示lambda的最小值. Defaults to 0.0.
            hi (Optional[float], optional): 浮点数，表示lambda的最大值. Defaults to 1..
            max_iters (Optional[int], optional): 最大迭代次数. Defaults to 1000..
            auto_step (Optional[bool], optional): 是否自动增加迭代次数，默认为False. Defaults to False.
        """
        super(WarmStartGradientReverseLayer, self).__init__()  # 调用父类的初始化方法
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """定义层的输入和输出计算逻辑

        Args:
            input (torch.Tensor): 层的输入

        Returns:
            torch.Tensor: 层的输入
        """
        coeff = np.float32(
            2.0 * (self.hi - self.lo) / (1.0 * np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )  # 计算出lambda的值
        if self.auto_step:  # 如果自动增加迭代次数
            self.step()  # 调用step方法，增加迭代次数
        return GradientReverseFunction.apply(input, coeff)  # 调用GRL函数的apply方法，将输入和lambda的值传入，得到输出
    

    def step(self):  # 增加迭代次数的方法
        self.iter_num += 1