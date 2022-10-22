from collections import OrderedDict

import torch
from torch import nn
from torch.jit.annotations import Dict
from torch.nn import functional as F


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


# def _log_api_usage_once(obj: Any) -> None:
#     if not obj.__module__.startswith("torchvision"):
#         return
#     name = obj.__class__.__name__
#     if isinstance(obj, FunctionType):
#         name = obj.__name__
#     torch._C._log_api_usage_once(f"{obj.__module__}.{name}")


# class ConvNormActivation(torch.nn.Sequential):
#     """
#     Configurable block used for Convolution-Normalzation-Activation blocks.
#     Args:
#         in_channels (int): Number of channels in the input image
#         out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
#         kernel_size: (int, optional): Size of the convolving kernel. Default: 3
#         stride (int, optional): Stride of the convolution. Default: 1
#         padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#         norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolutiuon layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
#         activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
#         dilation (int): Spacing between kernel elements. Default: 1
#         inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
#         bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         padding: Optional[int] = None,
#         groups: int = 1,
#         norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#         dilation: int = 1,
#         inplace: bool = True,
#         bias: Optional[bool] = None,
#     ) -> None:
#         if padding is None:
#             padding = (kernel_size - 1) // 2 * dilation
#         if bias is None:
#             bias = norm_layer is None
#         layers = [
#             torch.nn.Conv2d(
#                 in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias,
#             )
#         ]
#         if norm_layer is not None:
#             layers.append(norm_layer(out_channels))
#         if activation_layer is not None:
#             layers.append(activation_layer(inplace=inplace))
#         super().__init__(*layers)
#         _log_api_usage_once(self)
#         self.out_channels = out_channels


# class SqueezeExcitation(torch.nn.Module):
#     """
#     This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
#     Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.
#     Args:
#         input_channels (int): Number of channels in the input image
#         squeeze_channels (int): Number of squeeze channels
#         activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
#         scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
#     """

#     def __init__(
#         self,
#         input_channels: int,
#         squeeze_channels: int,
#         activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
#         scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
#     ) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
#         self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
#         self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
#         self.activation = activation()
#         self.scale_activation = scale_activation()

#     def _scale(self, input: Tensor) -> Tensor:
#         scale = self.avgpool(input)
#         scale = self.fc1(scale)
#         scale = self.activation(scale)
#         scale = self.fc2(scale)
#         return self.scale_activation(scale)

#     def forward(self, input: Tensor) -> Tensor:
#         scale = self._scale(input)
#         return scale * input


# def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
#     """
#     Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
#     <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
#     branches of residual architectures.
#     Args:
#         input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
#                     being its batch i.e. a batch with ``N`` rows.
#         p (float): probability of the input to be zeroed.
#         mode (str): ``"batch"`` or ``"row"``.
#                     ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
#                     randomly selected rows from the batch.
#         training: apply stochastic depth if is ``True``. Default: ``True``
#     Returns:
#         Tensor[N, ...]: The randomly zeroed tensor.
#     """
#     if not torch.jit.is_scripting() and not torch.jit.is_tracing():
#         _log_api_usage_once(stochastic_depth)
#     if p < 0.0 or p > 1.0:
#         raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
#     if mode not in ["batch", "row"]:
#         raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
#     if not training or p == 0.0:
#         return input

#     survival_rate = 1.0 - p
#     if mode == "row":
#         size = [input.shape[0]] + [1] * (input.ndim - 1)
#     else:
#         size = [1] * input.ndim
#     noise = torch.empty(size, dtype=input.dtype, device=input.device)
#     noise = noise.bernoulli_(survival_rate)
#     if survival_rate > 0.0:
#         noise.div_(survival_rate)
#     return input * noise


# torch.fx.wrap("stochastic_depth")


# class StochasticDepth(nn.Module):
#     """
#     See :func:`stochastic_depth`.
#     """

#     def __init__(self, p: float, mode: str) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         self.p = p
#         self.mode = mode

#     def forward(self, input: Tensor) -> Tensor:
#         return stochastic_depth(input, self.p, self.mode, self.training)

#     def __repr__(self) -> str:
#         tmpstr = self.__class__.__name__ + "("
#         tmpstr += "p=" + str(self.p)
#         tmpstr += ", mode=" + str(self.mode)
#         tmpstr += ")"
#         return tmpstr
