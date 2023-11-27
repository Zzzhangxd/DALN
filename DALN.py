import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function
from torchvision.models import resnet50, ResNet50_Weights

# GradientReverseFunction 和 GradientReverseLayer 的定义
class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, input, coeff=1.0):
        ctx.coeff = coeff
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

# WarmStartGradientReverseLayer 的定义
class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input):
        alpha_tensor = torch.tensor(self.alpha, device=input.device)
        iter_num_tensor = torch.tensor(self.iter_num, device=input.device)
        max_iters_tensor = torch.tensor(self.max_iters, device=input.device)

        coeff = 2.0 * (self.hi - self.lo) / (1.0 + torch.exp(-alpha_tensor * iter_num_tensor / max_iters_tensor)) - (self.hi - self.lo) + self.lo

        if self.auto_step:
            self.step()

        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1

# NuclearWassersteinDiscrepancy 的定义
class NuclearWassersteinDiscrepancy(nn.Module):
    def __init__(self, classifier):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True)
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s, y_t):
        pre_s, pre_t = torch.softmax(y_s, dim=1), torch.softmax(y_t, dim=1)
        loss = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / y_t.shape[0]
        return loss

    def forward(self, f):
        f_grl = self.grl(f)
        y = self.classifier(f_grl)
        y_s, y_t = y.chunk(2, dim=0)
        loss = self.n_discrepancy(y_s, y_t)
        return loss

# FeatureExtractor 和 Classifier 的定义
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        return self.fc(x)

# DomainAdaptationModel 的定义
class DomainAdaptationModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.grl_layer = WarmStartGradientReverseLayer()
        self.nwd_loss = NuclearWassersteinDiscrepancy(self.classifier)

    def forward(self, x_s, x_t, labels_s):
        f_s = self.feature_extractor(x_s)
        f_t = self.feature_extractor(x_t)
        preds_s = self.classifier(f_s)
        class_loss = nn.CrossEntropyLoss()(preds_s, labels_s)
        f_combined = torch.cat([f_s, f_t], dim=0)
        domain_loss = self.nwd_loss(f_combined)
        return class_loss + domain_loss
