### COPY of https://github.com/samirak98/FourierImaging/blob/main/fourierimaging/modules/trigoInterpolation.py

from collections.abc import Iterable
import math
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as tf
from typing_extensions import override


def rfftshift(x: Tensor) -> Tensor:
    """Returns a shifted version of a matrix obtained by torch.fft.rfft2(x), i.e. the left half of torch.fft.fftshift(torch.fft.fft2(x))"""
    # Real FT shifting thus consists of two steps: 1d-Shifting the last dimension and flipping in second to last dimension and taking the complex conjugate,
    # to get the frequency order [-N,...,-1,0]
    return torch.conj(torch.flip(fft.fftshift(x, dim=-2), dims=[-1]))


def irfftshift(x: Tensor) -> Tensor:
    """Inverts rfftshift: Returns a non-shifted version of the left half of torch.fft.fftshift(torch.fft.fft2(x)), i.e. torch.fft.rfft2(x)"""
    return torch.conj(torch.flip(fft.ifftshift(x, dim=-2), dims=[-1]))


def symmetric_padding(xf_old: Tensor, im_shape_old: npt.NDArray[np.int_], im_shape_new: npt.NDArray[np.int_]) -> Tensor:
    """Returns the rfft2 of the real trigonometric interpolation of an image x of shape 'im_shape_old' and shifted real Fourier transform 'xf_old' to shape 'im_shape_new'"""
    # xf_old has to be a shifted rfft2 matrix
    add_shape = np.array(xf_old.shape[:-2])  # [batchsize, channels]

    ft_height_old = im_shape_old[-2] + (1 - im_shape_old[-2] % 2)  # always odd
    ft_width_old = im_shape_old[-1] // 2 + 1
    ft_height_new = im_shape_new[-2] + (1 - im_shape_new[-2] % 2)  # always odd
    ft_width_new = im_shape_new[-1] // 2 + 1
    xf_shape = tuple(add_shape) + (ft_height_old, ft_width_old)  # shape for the unpadded trafo array

    pad_height = (ft_height_new - ft_height_old) // 2  # the difference between both ft shapes is always even
    pad_width = ft_width_new - ft_width_old
    pad_list = [pad_width, 0, pad_height, pad_height]  #'starting from the last dimension and moving forward, (padding_left,padding_right, padding_top,padding_bottom)'
    xf = torch.zeros(size=xf_shape, dtype=torch.cfloat, device=xf_old.device)
    xf[..., : im_shape_old[-2], :] = xf_old  # for odd dimensions, this already represents a set of Herm.sym. coefficients

    # for even dimensions, the coefficients corresponding to the nyquist frequency are split symmetrically
    if im_shape_old[-2] % 2 == 0:
        xf[..., 0, :] *= 0.5  # nyquist_row /2
        xf[..., -1, :] = xf[..., 0, :]  # nyquist_row/2

    if im_shape_old[-1] % 2 == 0:
        xf[..., :, 0] *= 0.5  # nyquist_col/2

    ## Trigonometric interpolation: Zero-Padding/Cut-Off of the symmetric (odd-dimensional) Fourier transform, if needed convert to even dimensions
    # Zero-padding/cut-off to odd_dimension such that desired_dimension <= odd_dimension <= desired_dimension + 1
    xf_pad = tf.pad(xf, pad_list)

    # if desired dimension is even, reverse the nyquist splitting
    if im_shape_new[-2] % 2 == 0:
        xf_pad[..., 0, :] *= 2

    if im_shape_new[-1] % 2 == 0:
        xf_pad[..., :, 0] *= 2

    return xf_pad[..., : im_shape_new[-2], :]


# Complex multiplication (from https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py)
def compl_mul2d(input: Tensor, weight: Tensor) -> Tensor:
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", input, weight)


# Convolution with 0-padding of kernel in spectral domain
def spectral_conv2d(x: Tensor, kernel: Tensor, kernel_shape: npt.NDArray[np.int32], norm: Literal["forward", "backward", "ortho"], groups: int = 1) -> Tensor:
    assert kernel.shape[1] % groups == 0, "output shape, i.e., kernel.shape[1] = " + str(kernel.shape[1]) + " is not a multiple of groups = " + str(groups)
    in_size = kernel.shape[0]
    out_size = kernel.shape[1] // groups
    output = torch.concat(
        [spectral_conv2d_nogroup(x[:, i * in_size : (i + 1) * in_size], kernel[:, i * out_size : (i + 1) * out_size], kernel_shape, norm) for i in range(groups)], dim=1
    )

    return output


def spectral_conv2d_nogroup(x: Tensor, kernel: Tensor, kernel_shape: npt.NDArray[np.int32], norm: Literal["forward", "backward", "ortho"]) -> Tensor:
    x_shape = np.array(x.shape[-2:])

    compute_shape = np.min([x_shape, kernel_shape], axis=1)
    # adapt the weight parameters to input dimension by trigonometric interpolation (to odd dimension)
    multiplier_padded = symmetric_padding(kernel, kernel_shape, compute_shape + (1 - compute_shape % 2))

    # For even input dimensions we interpolate to the next higher odd dimension
    # this could be optimized by checking dimensions first
    x_ft_padded = symmetric_padding(rfftshift(fft.rfft2(x, norm=norm)), x_shape, compute_shape + (1 - compute_shape % 2))

    # Elementwise multiplication of rfft-coefficients and return to physical space after correcting dimension with symmetric padding if desired dimension is even
    output = fft.irfft2(irfftshift(symmetric_padding(compl_mul2d(x_ft_padded, multiplier_padded), compute_shape + (1 - compute_shape % 2), x_shape)), norm=norm, s=tuple(x_shape))

    return output


class TrigonometricResize_2d(nn.Module):
    """Resize 2d image with trigonometric interpolation"""

    def __init__(
        self,
        shape: npt.NDArray[np.int_] | tuple[int, ...],
        norm: Literal["forward", "backward", "ortho"] = "forward",
        upsample: bool = True,
        downsample: bool = True,
        check_comp: bool = False,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.norm = norm
        self.upsample = upsample  # upsample == False: don't reshape if input shape is smaller than self.shape (downsampling only)
        self.downsample = downsample  # downsample == False: don't reshape if input shape is larger than self.shape (upsampling only)
        self.check_comp = check_comp

    @override
    def __call__(self, x: Tensor, keep_shape: int = None) -> Tensor | tuple[Tensor, bool]:
        if self.upsample and self.downsample: # this is the standard case
            im_shape_new = np.array(self.shape)
        elif not self.upsample: # this is the special case for contracting path (down) of spectral UNet 
            im_shape_new = np.minimum(np.array(self.shape), np.array(x.shape[-2:])) # only downsampling
            keep_shape = x.shape[-1] # remember which size came in for upsampling 
        elif not self.downsample: # this is the special case for expanding path (up) of spectral UNet
            if keep_shape is None:
                im_shape_new = np.maximum(np.array(self.shape), np.array(x.shape[-2:]))
            else:
                im_shape_new = np.array([keep_shape, keep_shape]) # upsample to remembered size
        else:
            raise NotImplementedError("Either upsampling or downsampling has to be True")

        if torch.is_complex(x):  # for complex valued functions, trigonometric interpolations is done by simple zero-padding of the Fourier coefficients
            x_inter = fft.ifft2(fft.fft2(x, norm=self.norm), s=self.shape, norm=self.norm)
        else:  # for real valued functions, the coefficients have to be Hermitian symmetric
            im_shape_old = np.array(x.shape[-2:])  # this has to be saved since it cannot be uniquely obtained by rfft(x)
            try:
                x_inter = fft.irfft2(irfftshift(symmetric_padding(rfftshift(fft.rfft2(x, norm=self.norm)), im_shape_old, im_shape_new)), s=tuple(im_shape_new), norm=self.norm)
            except RuntimeError as e:
                print(im_shape_old)
                print(im_shape_new)
                raise e
        if not self.upsample:
            return x_inter, keep_shape
        else:
            return x_inter

    def check_symmetry(self, x: Tensor, im_shape: npt.NDArray[np.int_] | None = None, threshold: float = 1e-5) -> None:
        ## Helper function to check Hermitian symmetry of an odd-dimensioned matrix of Fourier coefficients.
        # Symmetry is fulfilled iff the coefficients correspond to a function which attains only real values at the considered sampling points
        if self.check_comp:
            x_flip = torch.flip(x, dims=[-2, -1])
            symmetry_check = x - torch.conj(x_flip)
            symmetry_norm = torch.norm(symmetry_check, p=float("Inf"))
            if symmetry_norm > threshold:
                print("Not symmetric: (norm: " + str(symmetry_norm) + " for old shape: " + str(im_shape))

    def check_imag(self, x: Tensor, im_shape: npt.NDArray[np.int_] | None = None, threshold: float = 1e-5) -> None:
        if self.check_comp:
            imag_norm = torch.norm(x.imag, p=float("Inf"))
            if imag_norm > threshold:
                print("The imaginary part of the image is unusual high, norm: " + str(imag_norm) + " for old shape: " + str(im_shape))


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ksize1: int = 1, ksize2: int = 1, norm: Literal["forward", "backward", "ortho"] = "forward", bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm: Literal["forward", "backward", "ortho"] = norm

        scale = 1 / (in_channels * out_channels)

        weight = torch.rand(in_channels, out_channels, ksize1, ksize2 // 2 + 1, dtype=torch.cfloat)
        weight = scale * (weight)

        self.odd = ksize2 % 2
        weight[:, :, 0, 0].imag = cast(Tensor, 0.0)
        self.ksize1 = weight.shape[-2]
        self.ksize2 = 2 * (weight.shape[-1] - 1) + self.odd
        self.weight = nn.Parameter(weight.clone())

        self.bias = nn.Parameter(torch.empty((out_channels,))) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # pyright: ignore [reportPrivateUsage]
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    @override
    def forward(self, x: Tensor) -> Tensor:
        kernel_shape = np.array([self.ksize1, self.ksize2])
        output = spectral_conv2d(x, self.weight, kernel_shape, norm=self.norm)
        if self.bias is not None:
            output = output + self.bias[..., None, None]

        return output

    @override
    def extra_repr(self):
        s = "{in_channels}, {out_channels}, {ksize1}, {ksize2}"
        return s.format(**self.__dict__)


def spatial_to_spectral(
    weight: Tensor, im_shape: Iterable[int], norm: Literal["forward", "backward", "ortho"] = "forward", conv_like_cnn: bool = True, ksize: npt.NDArray[np.int32] | None = None
) -> Tensor:
    im_shape = np.array(im_shape)

    weight = torch.flip(weight, dims=[-2, -1])
    weight = torch.permute(weight, (1, 0, 2, 3))  # torch.conv2d performs a cross-correlation, i.e., convolution with flipped weight
    if norm == "forward":
        weight *= np.prod(im_shape)
    elif norm == "ortho":
        weight *= np.sqrt(np.prod(im_shape))

    kernel_shape = np.array([weight.shape[-2], weight.shape[-1]])
    shape_diff = im_shape - kernel_shape
    pad = np.sign(shape_diff) * np.abs(shape_diff) // 2
    odd_bias = np.abs(shape_diff) % 2
    oddity_old = kernel_shape % 2
    pad_list = [
        pad[-1] + odd_bias[-1] * oddity_old[-1],
        pad[-1] + odd_bias[-1] * (1 - oddity_old[-1]),
        pad[-2] + odd_bias[-2] * oddity_old[-2],
        pad[-2] + odd_bias[-2] * (1 - oddity_old[-2]),
    ]  # starting from the last dimension and moving forward, (padding_left,padding_right, padding_top,padding_bottom)'

    spectral_weight = rfftshift(fft.rfft2(fft.ifftshift(tf.pad(weight, pad_list), dim=[-2, -1]), norm=norm))

    # the discrete approximation of continuous convolution in spatial domain differs from the spectral implementation for even dimensions, if conv_like_cnn, we use spatial approach
    if conv_like_cnn:
        if im_shape[-2] % 2 == 0:
            spectral_weight[..., 0, :] *= 2
        if im_shape[-1] % 2 == 0:
            spectral_weight[..., :, 0] *= 2

    if ksize is not None:
        ksize = np.array(ksize)

        spectral_weight = symmetric_padding(spectral_weight, im_shape, ksize)
    return spectral_weight


def gen_from_Conv2d(conv: nn.Module, ksize1: int, ksize2: int, norm: Literal["forward", "backward", "ortho"] = "forward") -> SpectralConv2d:
    spatial_weight = conv.weight
    in_channels = conv.weight.shape[1]
    out_channels = conv.weight.shape[0]
    bias = conv.bias is not None
    spec_conv = SpectralConv2d(in_channels, out_channels, ksize1=ksize1, ksize2=ksize2, bias=bias)
    spec_conv.weight = nn.Parameter(spatial_to_spectral(spatial_weight, im_shape=(ksize1, ksize2), conv_like_cnn=True))
    spec_conv.bias = nn.Parameter(conv.bias.clone() if conv.bias is not None else torch.empty((out_channels,)))
    return spec_conv


class Residual_Layer(nn.Module):
    def __init__(self, linear_layer: nn.Module, activation_function: nn.Module) -> None:
        super().__init__()
        self.linear_layer = linear_layer
        self.activation_function = activation_function

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.activation_function(self.linear_layer(x))
