class depth_separable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int) -> None:
        super(depth_separable, self).__init__()
        self.w_depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, k),
            stride=1,
            groups=in_channels
        )
        self.h_depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(k, 1),
            stride=1,
            groups=in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=(k - 1) // 2,
            groups=1
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        return self.act(self.point_conv(self.bn(self.h_depth_conv(self.w_depth_conv(x)))))

class MELK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MELK, self).__init__()
        assert in_channels % 4 == 0
        self.outc = int(in_channels * 0.25)
        self.d1 = Conv(in_channels, self.outc, 3, 1, 1)
        self.d2 = depth_separable(in_channels, self.outc, 5)
        self.d3 = depth_separable(in_channels, self.outc, 7)
        self.d4 = depth_separable(in_channels, self.outc, 9)
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.silu = nn.SiLU()
        self.end = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False, groups=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.view(b, c, h * w).permute(0, 2, 1)
        q = self.query(y)
        k = self.key(y)
        v = self.value(y)
        scores = torch.matmul(q.transpose(-2, -1), k)
        scores = F.softmax(scores, dim=-1)
        qkv = torch.matmul(v, scores)
        qkv = qkv.view(b, c, h, w) + x
        p1 = self.d1(x)
        p2 = self.d2(x)
        p3 = self.d3(x)
        p4 = self.d4(x)
        p_out = torch.cat([p1, p2, p3, p4], 1)
        end = p_out + qkv
        end = self.end(end)
        return end