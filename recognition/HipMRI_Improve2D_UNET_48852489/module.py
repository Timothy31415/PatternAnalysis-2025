# module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- small helpers ---------
def conv3x3(cin, cout, s=1):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=s, padding=1, bias=False)

def conv1x1(cin, cout, s=1):
    return nn.Conv2d(cin, cout, kernel_size=1, stride=s, bias=False)

# --------- building blocks ---------
class PreActResBlock2D(nn.Module):
    """
    Pre-activation residual block:
      x -> IN -> LeakyReLU -> Conv3 -> Drop -> IN -> LeakyReLU -> Conv3 -> +x
    """
    def __init__(self, c: int, pdrop: float = 0.3):
        super().__init__()
        self.in1 = nn.InstanceNorm2d(c, affine=True)
        self.in2 = nn.InstanceNorm2d(c, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.conv1 = conv3x3(c, c)
        self.drop = nn.Dropout2d(pdrop) if pdrop > 0 else nn.Identity()
        self.conv2 = conv3x3(c, c)

    def forward(self, x):
        y = self.conv1(self.act(self.in1(x)))
        y = self.drop(y)
        y = self.conv2(self.act(self.in2(y)))
        return x + y

class LocalizationBlock2D(nn.Module):
    """
    After concat(skip, up): Conv3 -> IN+LeakyReLU -> 1x1 channel reduce
    """
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = conv3x3(cin, cin)
        self.in1  = nn.InstanceNorm2d(cin, affine=True)
        self.act  = nn.LeakyReLU(0.01, inplace=True)
        self.reduce = conv1x1(cin, cout)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(self.in1(x))
        x = self.reduce(x)
        return x

# --------- the model ---------
class ImprovedUNet2D(nn.Module):
    """
    Improved U-Net 2D (Isensee-ish):
      - Encoder: preact-res blocks, strided 3x3 down
      - Decoder: nearest upsample + 3x3 conv
      - Localization blocks (concat compression)
      - Deep supervision: sum of multi-scale heads (optional)
    """
    def __init__(
        self,
        in_channels: int = 1,     # e.g., T2 only => 1
        num_classes: int = 6,     # matches your loader default
        base: int = 32,           # feature width
        deep_supervision: bool = True,
        pdrop: float = 0.3
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Encoder
        self.stem = conv3x3(in_channels, base)     # (b, H, W)
        self.ctx1 = PreActResBlock2D(base, pdrop)
        self.down1 = conv3x3(base, base*2, s=2)    # (b*2, H/2, W/2)

        self.ctx2 = PreActResBlock2D(base*2, pdrop)
        self.down2 = conv3x3(base*2, base*4, s=2)  # (b*4, H/4, W/4)

        self.ctx3 = PreActResBlock2D(base*4, pdrop)
        self.down3 = conv3x3(base*4, base*8, s=2)  # (b*8, H/8, W/8)

        self.ctx4 = PreActResBlock2D(base*8, pdrop)  # bottleneck

        # Decoder
        self.up3  = conv3x3(base*8, base*8)
        self.loc3 = LocalizationBlock2D(base*8 + base*4, base*4)

        self.up2  = conv3x3(base*4, base*4)
        self.loc2 = LocalizationBlock2D(base*4 + base*2, base*2)

        self.up1  = conv3x3(base*2, base*2)
        self.loc1 = LocalizationBlock2D(base*2 + base, base)

        # Heads (deep supervision)
        self.head3 = conv1x1(base*4, num_classes)
        self.head2 = conv1x1(base*2, num_classes)
        self.head1 = conv1x1(base,   num_classes)  # final

    @staticmethod
    def _upsample(x, scale=2):
        return F.interpolate(x, scale_factor=scale, mode="nearest")

    def forward(self, x):
        # x: (B, C=in_channels, H, W) e.g. (B,1,256,128)
        s0 = self.stem(x)
        e1 = self.ctx1(s0)
        d1 = self.down1(e1)

        e2 = self.ctx2(d1)
        d2 = self.down2(e2)

        e3 = self.ctx3(d2)
        d3 = self.down3(e3)

        bott = self.ctx4(d3)

        # up from bottleneck (H/8 -> H/4)
        u3 = self._upsample(bott)
        u3 = self.up3(u3)
        cat3 = torch.cat([u3, e3], dim=1)
        l3 = self.loc3(cat3)

        # H/4 -> H/2
        u2 = self._upsample(l3)
        u2 = self.up2(u2)
        cat2 = torch.cat([u2, e2], dim=1)
        l2 = self.loc2(cat2)

        # H/2 -> H
        u1 = self._upsample(l2)
        u1 = self.up1(u1)
        cat1 = torch.cat([u1, e1], dim=1)
        l1 = self.loc1(cat1)

        logits1 = self.head1(l1)  # (B, K, H, W)
        if not self.deep_supervision:
            return logits1

        logits2 = self.head2(l2)  # (B, K, H/2, W/2)
        logits3 = self.head3(l3)  # (B, K, H/4, W/4)

        # upsample aux to full-res and sum
        logits2 = F.interpolate(logits2, size=logits1.shape[-2:], mode="bilinear", align_corners=False)
        logits3 = F.interpolate(logits3, size=logits1.shape[-2:], mode="bilinear", align_corners=False)
        return logits1 + logits2 + logits3

# --------- (optional) losses/metrics you can import in train.py ---------
class DiceLoss(nn.Module):
    """Multi-class soft Dice loss (expects logits BxKxHxW and targets BxHxW indices)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)           # BxKxHxW
        K = probs.shape[1]
        onehot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1.0)
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)             # K
        denom = (probs + onehot).sum(dims)             # K
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()

@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, ignore_bg: bool = False) -> float:
    probs = torch.softmax(logits, dim=1)
    pred  = probs.argmax(dim=1)  # BxHxW
    K = probs.shape[1]
    dices = []
    for k in range(K):
        if ignore_bg and k == 0: 
            continue
        pk = (pred == k).float()
        tk = (target == k).float()
        inter = (pk * tk).sum()
        denom = pk.sum() + tk.sum()
        d = (2 * inter + 1e-6) / (denom + 1e-6)
        dices.append(d.item())
    return float(sum(dices) / max(len(dices), 1))

# --------- quick self-test ---------
if __name__ == "__main__":
    B, C, H, W = 2, 1, 256, 128
    K = 6
    x = torch.randn(B, C, H, W)
    net = ImprovedUNet2D(in_channels=C, num_classes=K, base=32, deep_supervision=True)
    y = net(x)
    print("Logits shape:", tuple(y.shape))  # (2, 6, 256, 128)
