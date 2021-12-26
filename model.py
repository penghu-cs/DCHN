import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

class Dense_Net(nn.Module):
    def __init__(self, input_dim=28*28, out_dim=20, norm=True):
        super(Dense_Net, self).__init__()
        mid_num1, mid_num2 = 4096, 4096#8172
        self.hash_layer = nn.Sequential(
            nn.Linear(input_dim, mid_num1),
            nn.ReLU(),
            nn.Linear(mid_num1, mid_num2),
            nn.ReLU(),
            nn.Linear(mid_num2, out_dim),
        )
        self.norm = norm

    def forward(self, x):
        out4 = self.hash_layer(x)
        if self.norm:
            norm_x = torch.norm(out4, dim=1, keepdim=True)
            out4 = out4 / norm_x
        return [out4]

class ImgNet(nn.Module):
    def __init__(self, out_dim, norm=True):
        super(ImgNet, self).__init__()
        self.vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
        self.vgg19_bn.classifier = nn.Sequential(*list(self.vgg19_bn.classifier.children())[:-2])
        mid_num = 4096
        self.hash_layer = nn.Sequential(
            nn.Linear(4096, mid_num),
            nn.ReLU(),
            nn.Linear(4096, mid_num),
            nn.ReLU(),
            nn.Linear(4096, out_dim)
        )
        self.norm = norm

    def forward(self, x, finetune=True):
        if finetune:
            x = self.vgg19_bn.features(x)
            x = x.view(x.size(0), -1)
            feat = self.vgg19_bn.classifier(x)
        else:
            self.vgg19_bn = self.vgg19_bn.eval()
            with torch.no_grad():
                x = self.vgg19_bn.features(x)
                x = x.view(x.size(0), -1)
                feat = self.vgg19_bn.classifier(x)
        feat = self.hash_layer(feat)
        if self.norm:
            norm_x = torch.norm(feat, dim=1, keepdim=True)
            out4 = feat / norm_x
            return [feat, out4]
        else:
            return [feat, feat]
