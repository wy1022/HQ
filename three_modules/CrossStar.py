class CrossStar(nn.Module):
    def __init__(self, c1, c2, mlp_ratio=2):
        super(CrossStar, self).__init__()
        self.f1 = nn.Conv2d(c1, mlp_ratio * c1, 1, 1, 0)
        self.f2 = nn.Conv2d(c1, mlp_ratio * c1, 1, 1, 0)
        self.f3 = nn.Conv2d(c1, mlp_ratio * c1, 3, 1, 1)
        self.f4 = nn.Conv2d(c1, mlp_ratio * c1, 3, 1, 1)
        self.g = nn.Conv2d(int(0.5 * mlp_ratio * c1), c2, 1)
        self.conv1_1 = nn.Conv2d(c1, int(0.5 * mlp_ratio * c1), 1, 1, 0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.GELU()
        self.act2 = nn.SiLU()

    def forward(self, x):
        x1, x2, x3, x4 = self.f1(x), self.f2(x), self.f3(x), self.f4(x)
        x1 = self.act(x1)
        x4 = self.act2(x4)
        x11, x12 = torch.chunk(x1, 2, 1)
        x21, x22 = torch.chunk(x2, 2, 1)
        x31, x32 = torch.chunk(x3, 2, 1)
        x41, x42 = torch.chunk(x4, 2, 1)
        y1 = x11 * x21
        y2 = x31 * x41
        y3 = x12 * x32
        y4 = x22 * x42
        y = y1 + y2 + y3 + y4 + self.conv1_1(x)
        y = self.g(y)
        x = self.maxpool(y)
        return x