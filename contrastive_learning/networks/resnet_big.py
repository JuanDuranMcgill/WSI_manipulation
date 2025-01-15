"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class SupConEmbeddingNet(nn.Module):
    """projection head for embeddings"""
    def __init__(self, input_dim=192, head='mlp', feat_dim=192):
        super(SupConEmbeddingNet, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(input_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):


        #print(f"Input shape: {x.shape}")

        feat = F.normalize(self.head(x), dim=1)
        #feat = self.head(x)
        return feat
    
class EnhancedSupConEmbeddingNet(nn.Module):
    """Enhanced projection head for embeddings"""
    def __init__(self, input_dim=192, feat_dim=192):
        super(EnhancedSupConEmbeddingNet, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),   # Increase hidden layer size
            nn.GELU(),                   # Use GELU activation for smoother transitions
            nn.Linear(256, 256),         # Additional layer for more complexity
            nn.GELU(),
            nn.Linear(256, feat_dim),    # Output layer with desired feature dimension
            nn.BatchNorm1d(feat_dim)     # Apply batch normalization before output
        )

    def forward(self, x):
        feat = self.head(x)
        feat = F.normalize(feat, dim=1)  # Normalize after the final projection
        return feat

class SupConEmbeddingNet_n(nn.Module):
    """Projection head for embeddings with support for sequence input"""
    def __init__(self, input_dim=192, seq_len=100, head='mlp', feat_dim=192):
        super(SupConEmbeddingNet_n, self).__init__()  # Corrected super() call
        self.seq_len = seq_len  # Length of the sequence (max_n)
        
        if head == 'linear':
            self.head = nn.Linear(input_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, input_dim)

        print(f"Input shape: {x.shape}")

        batch_size, seq_len, _ = x.size()
        
        # Apply the projection head to each vector in the sequence
        x = x.view(-1, x.size(-1))  # Flatten the seq_len dimension with the batch dimension
        feat = self.head(x)
        feat = F.normalize(feat, dim=1)
        
        # Reshape back to (batch_size, seq_len, feat_dim)
        feat = feat.view(batch_size, seq_len, -1)
        
        print(f"Input shape: {feat.shape}")
        return feat
    
class EnhancedSupConEmbeddingNet_n(nn.Module):
    """Enhanced projection head for sequence input embeddings"""
    def __init__(self, input_dim=192, seq_len=100, feat_dim=192):
        super(EnhancedSupConEmbeddingNet_n, self).__init__()
        self.seq_len = seq_len  # Length of the sequence (max_n)

        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),   # Increase hidden layer size
            nn.GELU(),                   # Use GELU activation for smoother transitions
            nn.Linear(256, 256),         # Additional layer for more complexity
            nn.GELU(),
            nn.Linear(256, feat_dim),    # Output layer with desired feature dimension
            nn.BatchNorm1d(feat_dim)     # Apply batch normalization before output
        )

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, input_dim)

        print(f"Input shape: {x.shape}")

        batch_size, seq_len, _ = x.size()
        
        # Apply the enhanced projection head to each vector in the sequence
        x = x.view(-1, x.size(-1))  # Flatten the seq_len dimension with the batch dimension
        feat = self.head(x)
        feat = F.normalize(feat, dim=1)  # Normalize after the final projection
        
        # Reshape back to (batch_size, seq_len, feat_dim)
        feat = feat.view(batch_size, seq_len, -1)
        
        print(f"Output shape: {feat.shape}")
        return feat
    
class SupConEmbeddingNetWithAttention(nn.Module):
    """projection head for embeddings with self-attention"""
    def __init__(self, input_dim=192, hidden_dim=256, feat_dim=128):
        super(SupConEmbeddingNetWithAttention, self).__init__()
        self.attention = SelfAttention(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, feat_dim)
        )

    def forward(self, x):
        # x is of shape (n, 192)
        attn_output = self.attention(x)  # output shape will be (1, 192)
        feat = F.normalize(self.head(attn_output), dim=1)  # feed into the projection head
        return feat

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        Q = self.query(x)  # (n, hidden_dim)
        K = self.key(x)  # (n, hidden_dim)
        V = self.value(x)  # (n, hidden_dim)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # (n, n)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (n, n)
        
        attn_output = torch.matmul(attn_weights, V)  # (n, hidden_dim)
        output = self.fc(attn_output.sum(dim=0))  # (hidden_dim) -> (input_dim)
        
        return output.unsqueeze(0)  # (1, input_dim)
    
class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
