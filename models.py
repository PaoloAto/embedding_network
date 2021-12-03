import torch
import torch.nn as nn

import math

class PIF(nn.Module):

    def __init__(self, in_channels=2048, out_channels=340, kernel_size=1, padding=10, upscale=2, dropout=0.0, inplace=True):
        super().__init__()

        self.dropout = torch.nn.Dropout2d(p=dropout)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=1)
        self.upsample = torch.nn.PixelShuffle(upscale)


    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        x = self.conv(x)
        # upscale
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                # negative axes not supported by ONNX TensorRT
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                # the int() forces the tracer to use static shape
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]

        # Extract some shape parameters once.
        # Convert to int so that shape is constant in ONNX export.
        x_size = x.size()
        batch_size = x_size[0]
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        x = x.view(
            batch_size,
            self.meta.n_fields,
            self.meta.n_confidences + self.meta.n_vectors * 3 + self.meta.n_scales,
            feature_height,
            feature_width
        )

        # if not self.training and self.inplace_ops:
        #     # classification
        #     classes_x = x[:, :, 0:self.meta.n_confidences]
        #     torch.sigmoid_(classes_x)

        #     # regressions x: add index
        #     if self.meta.n_vectors > 0:
        #         index_field = index_field_torch((feature_height, feature_width), device=x.device)
        #         first_reg_feature = self.meta.n_confidences
        #         for i, do_offset in enumerate(self.meta.vector_offsets):
        #             if not do_offset:
        #                 continue
        #             reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
        #             reg_x.add_(index_field)

        #     # scale
        #     first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
        #     scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
        #     scales_x[:] = torch.nn.functional.softplus(scales_x)
        # elif not self.training and not self.inplace_ops:
        #     # TODO: CoreMLv4 does not like strided slices.
        #     # Strides are avoided when switching the first and second dim
        #     # temporarily.
        #     x = torch.transpose(x, 1, 2)

        #     # classification
        #     classes_x = x[:, 0:self.meta.n_confidences]
        #     classes_x = torch.sigmoid(classes_x)

        #     # regressions x
        #     first_reg_feature = self.meta.n_confidences
        #     regs_x = [
        #         x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
        #         for i in range(self.meta.n_vectors)
        #     ]
        #     # regressions x: add index
        #     index_field = index_field_torch(
        #         (feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
        #     # TODO: coreml export does not work with the index_field creation in the graph.
        #     index_field = torch.from_numpy(index_field.numpy())
        #     regs_x = [reg_x + index_field if do_offset else reg_x
        #               for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]

        #     # regressions logb
        #     first_reglogb_feature = self.meta.n_confidences + self.meta.n_vectors * 2
        #     regs_logb = x[:, first_reglogb_feature:first_reglogb_feature + self.meta.n_vectors]

        #     # scale
        #     first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
        #     scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]
        #     scales_x = torch.nn.functional.softplus(scales_x)

        #     # concat
        #     x = torch.cat([classes_x, *regs_x, regs_logb, scales_x], dim=1)

        #     # TODO: CoreMLv4 problem (see above).
        #     x = torch.transpose(x, 1, 2)

        return x


class Resnet(nn.Module):
    pretrained = True
    pool0_stride = 0
    input_conv_stride = 2
    input_conv2_stride = 0
    remove_last_block = False
    block5_dilation = 1

    def __init__(self, torchvision_resnet=None, out_features=2048):
        super().__init__()

        if torchvision_resnet is None:
            from torchvision.models.resnet import resnet50 as torchvision_resnet
        modules = list(torchvision_resnet(self.pretrained).children())
        stride = 32

        input_modules = modules[:4]

        # input pool
        if self.pool0_stride:
            if self.pool0_stride != 2:
                # pylint: disable=protected-access
                input_modules[3].stride = torch.nn.modules.utils._pair(self.pool0_stride)
                stride = int(stride * 2 / self.pool0_stride)
        else:
            input_modules.pop(3)
            stride //= 2

        # input conv
        if self.input_conv_stride != 2:
            # pylint: disable=protected-access
            input_modules[0].stride = torch.nn.modules.utils._pair(self.input_conv_stride)
            stride = int(stride * 2 / self.input_conv_stride)

        # optional use a conv in place of the max pool
        if self.input_conv2_stride:
            assert not self.pool0_stride  # this is only intended as a replacement for maxpool
            channels = input_modules[0].out_channels
            conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU(inplace=True),
            )
            input_modules.append(conv2)
            stride *= 2

        # block 5
        block5 = modules[7]
        if self.remove_last_block:
            block5 = None
            stride //= 2
            out_features //= 2

        if self.block5_dilation != 1:
            stride //= 2
            for m in block5.modules():
                if not isinstance(m, torch.nn.Conv2d):
                    continue

                # also must be changed for the skip-conv that has kernel=1
                m.stride = torch.nn.modules.utils._pair(1)

                if m.kernel_size[0] == 1:
                    continue

                m.dilation = torch.nn.modules.utils._pair(self.block5_dilation)
                padding = (m.kernel_size[0] - 1) // 2 * self.block5_dilation

        self.input_block = torch.nn.Sequential(*input_modules)
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]
        self.block5 = block5

    def forward(self, x):
        x = self.input_block(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x