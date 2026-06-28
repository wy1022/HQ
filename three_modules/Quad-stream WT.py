def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class QSWT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, wt_levels=2, wt_type='db1'):
        super(QSWT, self).__init__()
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=True)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=True)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        self.wavelet_convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=kernel_size, padding=2, stride=1,
                          dilation=1, groups=in_channels * 4, bias=False),
                nn.BatchNorm2d(in_channels * 4),
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1, padding=0, stride=1,
                          dilation=1, groups=1, bias=False),
                nn.PReLU()
            ) for _ in range(self.wt_levels)]
        )
        self.wavelet_convs_lh = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3, padding=1, stride=1,
                          dilation=1, groups=in_channels * 4, bias=False),
                nn.BatchNorm2d(in_channels * 4),
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1, padding=0, stride=1,
                          dilation=1, groups=1, bias=False),
                nn.PReLU()
            ) for _ in range(self.wt_levels)]
        )
        self.wavelet_convs_hl = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3, padding=1, stride=1,
                          dilation=1, groups=in_channels * 4, bias=False),
                nn.BatchNorm2d(in_channels * 4),
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1, padding=0, stride=1,
                          dilation=1, groups=1, bias=False),
                nn.PReLU()
            ) for _ in range(self.wt_levels)]
        )
        self.wavelet_convs_hh = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1, padding=0, stride=1,
                          dilation=1, groups=1, bias=False),
                nn.BatchNorm2d(in_channels * 4),
                nn.PReLU()
            ) for _ in range(self.wt_levels)]
        )
        self.end = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False, groups=1),
            nn.GroupNorm(8, out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x
        curr_x_lh = x
        curr_x_hl = x
        curr_x_hh = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
                curr_x_lh = F.pad(curr_x_lh, curr_pads)
                curr_x_hl = F.pad(curr_x_hl, curr_pads)
                curr_x_hh = F.pad(curr_x_hh, curr_pads)
            curr_x = self.wt_function(curr_x_ll)
            curr_x2 = self.wt_function(curr_x_lh)
            curr_x3 = self.wt_function(curr_x_hl)
            curr_x4 = self.wt_function(curr_x_hh)
            curr_x_ll = curr_x[:, :, 0, :, :]
            curr_x_lh = curr_x2[:, :, 1, :, :]
            curr_x_hl = curr_x3[:, :, 2, :, :]
            curr_x_hh = curr_x4[:, :, 3, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag2 = curr_x2.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag3 = curr_x3.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag4 = curr_x4.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_convs[i](curr_x_tag)
            curr_x_tag2 = self.wavelet_convs_lh[i](curr_x_tag2)
            curr_x_tag3 = self.wavelet_convs_hl[i](curr_x_tag3)
            curr_x_tag4 = self.wavelet_convs_hh[i](curr_x_tag4)
            curr_x_tag = curr_x_tag.reshape(shape_x)
            curr_x_tag2 = curr_x_tag2.reshape(shape_x)
            curr_x_tag3 = curr_x_tag3.reshape(shape_x)
            curr_x_tag4 = curr_x_tag4.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h = torch.cat((curr_x_tag2[:, :, 1, :, :].unsqueeze(2), curr_x_tag3[:, :, 2, :, :].unsqueeze(2), curr_x_tag4[:, :, 3, :, :].unsqueeze(2)), 2)
            x_h_in_levels.append(x_h)
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        x = x + x_tag
        x = self.end(x)
        return x