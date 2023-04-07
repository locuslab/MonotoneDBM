import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiMonotoneHollowConv(nn.Module):
    def __init__(self, sizes, kernel_sizes, monotone_m=-1):
        """
        Hollow convolution A^T A X - blockdiag(A^T A) X

            sizes: list of (channel, w, h, groups) tuples
            kernel_sizes: 2d array of kernel_size values, 0 for no conv
            monotone_m: force layer to be strongly monotone with constant m, or not if m = -1
        """
        super().__init__()
        self.sizes = sizes
        self.m = monotone_m
        self.shapes = []
        self.add_module('biases', nn.ParameterList())
        self.g = nn.Parameter(torch.full((len(sizes),), 1.))
        self.norms = [None for _ in range(len(sizes))]
        self.groups = []
        for size in sizes:
            if len(size) == 4:
                c, h, w, group = size
                shape = (c, h, w)
                self.shapes.append(shape)
                self.biases.append(nn.Parameter(torch.rand(c, h, w)))
                self.groups.append(group)
            else:
                a, b = size
                self.shapes.append((a, 1, 1))
                self.biases.append(nn.Parameter(torch.rand(a, 1, 1)))
                self.groups.append(b)

        def padding(stride, kernel_size):
            """ Magic formula for padding"""

            def ceil_div(a, b):
                return -(a // -b)

            return ceil_div(stride * (kernel_size - 2) + kernel_size, 2)

        self.convs = nn.ModuleList([nn.ModuleList([None for _ in range(len(sizes))]) for _ in range(len(sizes))])
        for i in range(len(sizes)):
            for j in range(len(sizes)):
                if kernel_sizes[i, j] != 0:
                    if len(self.sizes[j]) == 4 and len(self.sizes[i]) == 4:
                        # conv -> conv
                        self.convs[i][j] = nn.Conv2d(sizes[j][0], sizes[i][0],
                                                     kernel_size=kernel_sizes[i, j],
                                                     stride=sizes[j][2] // sizes[i][2],
                                                     padding=padding(sizes[j][2] // sizes[i][2], kernel_sizes[i, j]),
                                                     bias=False)
                    elif len(self.sizes[j]) == 4 and len(self.sizes[i]) == 2:
                        # conv -> dense
                        self.convs[i][j] = nn.Conv2d(sizes[j][0], sizes[i][0],
                                                     kernel_size=(sizes[j][1], sizes[j][2]),
                                                     stride=1,
                                                     padding=0,
                                                     bias=False)
                    elif len(self.sizes[j]) == 2 and len(self.sizes[i]) == 2:
                        # dense -> dense
                        self.convs[i][j] = nn.Conv2d(sizes[j][0], sizes[i][0], 1, padding=0, stride=1, bias=False)

        self.max_strides = [max((self.convs[i][j].stride[0] if self.convs[i][j].padding[0] > 0
                                 else self.convs[i][j].kernel_size[0])
                                for i in range(len(sizes)) if self.convs[i][j] is not None)
                            for j in range(len(sizes))]

    def clean_norms(self):
        self.norms = [None for _ in range(len(self.sizes))]

    def tuple_to_tensor(self, z):
        # z_flatten_tuple is a tuple of variables of size bsz*variable_size
        bsz = z[0].shape[0]
        z_flatten_tuple = [z[i].view(bsz, -1) for i in range(len(z))]
        return torch.cat(z_flatten_tuple, 1)

    def tensor_to_tuple(self, z):
        z_tuple = []
        bsz = z.shape[0]
        curr_idx = 0
        for item in self.shapes:
            c, h, w = item
            size = c * h * w
            curr_z = z[:, curr_idx:curr_idx + size].view(bsz, c, h, w)

            z_tuple.append(curr_z)
            curr_idx += size
        return z_tuple

    def multiply(self, X):
        return [sum(F.conv2d(X[j], self.convs[i][j].weight,
                             padding=self.convs[i][j].padding, stride=self.convs[i][j].stride)
                    for j in range(len(X)) if self.convs[i][j] is not None) for i in range(len(X))]

    def multiply_transpose(self, X):
        """ Multiply A^T * X"""
        return [sum(F.conv_transpose2d(X[i], self.convs[i][j].weight,
                                       padding=self.convs[i][j].padding, stride=self.convs[i][j].stride,
                                       output_padding=(1 if self.convs[i][j].stride[0] > 1 and
                                                            self.convs[i][j].kernel_size[0] % 2 == 1 else 0))
                    for i in range(len(X)) if self.convs[i][j] is not None)
                for j in range(len(X))]

    def forward(self, *X):
        batch_size = X[0].shape[0]
        Y = self.multiply_transpose(self.multiply(X))

        # subtract off block diagonals
        for j in range(len(self.sizes)):
            max_stride = self.max_strides[j]
            groups = self.sizes[j][-1]
            n = self.sizes[j][0] // groups
            h, w = X[j].shape[2:4]
            blk_diag = torch.zeros(max_stride, max_stride, groups, n, n).to(X[0].device)

            for i in range(len(self.sizes)):
                if self.convs[i][j] is not None:
                    kernel_size = self.convs[i][j].kernel_size[0]
                    stride = self.convs[i][j].stride[0]

                    # compute diagonal components of A^T A per group

                    A0 = self.convs[i][j].weight.view(-1, groups, n, kernel_size, kernel_size)
                    A0 = A0.permute(1, 3, 4, 0, 2)
                    ATA = A0.transpose(3, 4) @ A0
                    # conv -> conv
                    if len(self.sizes[i]) == 4:
                        # magic formula to compute offsets for even/odd kernel sizes
                        if kernel_size % 2 == 0:
                            offset = (stride - (kernel_size // 2)) % stride
                        else:
                            offset = ((stride - kernel_size) // 2) % stride
                        for i1 in range(ATA.shape[1]):
                            for j1 in range(ATA.shape[2]):
                                blk_diag[(i1 + offset) % stride::stride, (j1 + offset) % stride::stride] += ATA[:, i1,
                                                                                                            j1]

                    elif len(self.sizes[i]) == 2:
                        # conv -> dense
                        for i1 in range(ATA.shape[1]):
                            for j1 in range(ATA.shape[2]):
                                blk_diag[i1, j1] += ATA[:, i1, j1]

            # subtract off tiled block diagonal component by groups
            X0 = X[j].view(batch_size, groups, n, h // max_stride, max_stride, w // max_stride, max_stride)
            Y_diag = (blk_diag @ X0.permute(3, 5, 4, 6, 1, 2, 0)).permute(6, 4, 5, 0, 2, 1, 3)
            Y0 = Y[j].view(batch_size, groups, n, h // max_stride, max_stride, w // max_stride, max_stride)
            Y0 -= Y_diag

            if self.m >= 0:
                if self.norms[j] is None:
                    norms = torch.linalg.norm(blk_diag, 2, (3, 4))
                    self.norms[j] = norms
                else:
                    norms = self.norms[j]
                scale = torch.clamp((1 - self.m) / norms, max=1.0)

                Y0 *= scale.permute(2, 0, 1)[None, :, None, None, :, None, :]
        # if self.m<0:
        for i, y in enumerate(Y):
            y -= self.biases[i]/self.biases[i].norm()*self.g[i]
        return tuple([-y for y in Y])
