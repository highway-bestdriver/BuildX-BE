from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Literal

class BaseLayer(BaseModel):
    type: str
    id: str
    input: Optional[Union[str, List[str]]] = None

class InputLayer(BaseLayer):
    type: Literal["Input"]
    shape: Optional[List[int]] = None

class Conv2dLayer(BaseLayer):
    type: Literal["Conv2d"]
    in_channels: int
    out_channels: int
    kernel_size: Union[int, List[int]]
    stride: Union[int, List[int]] = 1
    padding: Union[int, str] = 0
    dilation: Union[int, List[int]] = 1
    groups: int = 1
    bias: bool = True

class MaxPool2dLayer(BaseLayer):
    type: Literal["MaxPool2d"]
    kernel_size: Union[int, List[int]]
    stride: Optional[Union[int, List[int]]] = None
    padding: Union[int, List[int]] = 0
    dilation: Union[int, List[int]] = 1
    return_indices: bool = False
    ceil_mode: bool = False

class AvgPool2dLayer(BaseLayer):
    type: Literal["AvgPool2d"]
    kernel_size: Union[int, List[int]]
    stride: Optional[Union[int, List[int]]] = None
    padding: Union[int, List[int]] = 0
    ceil_mode: bool = False
    count_include_pad: bool = True
    divisor_override: Optional[int] = None

class AdaptiveAvgPool2dLayer(BaseLayer):
    type: Literal["AdaptiveAvgPool2d"]
    output_size: Union[int, List[int]]

class AdaptiveMaxPool2dLayer(BaseLayer):
    type: Literal["AdaptiveMaxPool2d"]
    output_size: Union[int, List[int]]

class LinearLayer(BaseLayer):
    type: Literal["Linear"]
    in_features: int
    out_features: int
    bias: bool = True

class DropoutLayer(BaseLayer):
    type: Literal["Dropout"]
    p: float = 0.5
    inplace: bool = False

class BatchNorm2dLayer(BaseLayer):
    type: Literal["BatchNorm2d"]
    num_features: int
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True

class FlattenLayer(BaseLayer):
    type: Literal["Flatten"]
    start_dim: int = 1
    end_dim: int = -1

class UpsampleLayer(BaseLayer):
    type: Literal["Upsample"]
    scale_factor: Union[float, List[float]]
    mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "area"]
    align_corners: Optional[bool] = None

class ConvTranspose2dLayer(BaseLayer):
    type: Literal["ConvTranspose2d"]
    in_channels: int
    out_channels: int
    kernel_size: Union[int, List[int]]
    stride: Union[int, List[int]] = 1
    padding: Union[int, List[int]] = 0
    output_padding: Union[int, List[int]] = 0
    groups: int = 1
    bias: bool = True
    dilation: Union[int, List[int]] = 1

class SequentialLayer(BaseLayer):
    type: Literal["Sequential"]
    layers: List[str]  # 하위 layer의 name/id 리스트

class IdentityLayer(BaseLayer):
    type: Literal["Identity"]

class ReLULayer(BaseLayer):
    type: Literal["ReLU"]
    inplace: bool = True

class LeakyReLULayer(BaseLayer):
    type: Literal["LeakyReLU"]
    negative_slope: float = 0.01
    inplace: bool = True

class SigmoidLayer(BaseLayer):
    type: Literal["Sigmoid"]

class TanhLayer(BaseLayer):
    type: Literal["Tanh"]

class SoftmaxLayer(BaseLayer):
    type: Literal["Softmax"]
    dim: int = 1

# === 통합 LayerUnion ===
LayerUnion = Union[
    InputLayer,
    Conv2dLayer,
    MaxPool2dLayer,
    AvgPool2dLayer,
    AdaptiveAvgPool2dLayer,
    AdaptiveMaxPool2dLayer,
    LinearLayer,
    DropoutLayer,
    BatchNorm2dLayer,
    FlattenLayer,
    UpsampleLayer,
    ConvTranspose2dLayer,
    SequentialLayer,
    IdentityLayer,
    ReLULayer,
    LeakyReLULayer,
    SigmoidLayer,
    TanhLayer,
    SoftmaxLayer
]

# === 공통 Base ===
class BaseTransform(BaseModel):
    type: str


# === Resize ===
class ResizeTransform(BaseTransform):
    type: Literal["Resize"]
    size: Optional[Union[int, List[int]]]
    interpolation: Optional[Union[int, str]] = "InterpolationMode.BILINEAR"
    max_size: Optional[int] = None
    antialias: Optional[bool] = True

# === CenterCrop ===
class CenterCropTransform(BaseTransform):
    type: Literal["CenterCrop"]
    size: Union[int, List[int]]

# === RandomCrop ===
class RandomCropTransform(BaseTransform):
    type: Literal["RandomCrop"]
    size: Union[int, List[int]]
    padding: Optional[Union[int, List[int]]] = None
    pad_if_needed: bool = False
    fill: Union[
        int,
        float,
        List[int],
        List[float],
        None,
        Dict[Union[type, str], Optional[Union[int, float, List[int], List[float]]]]
    ] = 0
    padding_mode: Literal["constant", "edge", "reflect", "symmetric"]

# === RandomHorizontalFlip ===
class RandomHorizontalFlipTransform(BaseTransform):
    type: Literal["RandomHorizontalFlip"]
    p: float = 0.5

# === RandomVerticalFlip ===
class RandomVerticalFlipTransform(BaseTransform):
    type: Literal["RandomVerticalFlip"]
    p: float = 0.5

# === RandomRotation ===
class RandomRotationTransform(BaseTransform):
    type: Literal["RandomRotation"]
    degrees: Union[float, List[float]]
    interpolation: Optional[Union[int, str]] = "InterpolationMode.NEAREST"
    expand: bool = False
    center: Optional[List[float]] = None
    fill: Union[
        int,
        float,
        List[int],
        List[float],
        None,
        Dict[Union[type, str], Optional[Union[int, float, List[int], List[float]]]]
    ] = 0

# === ColorJitter ===
class ColorJitterTransform(BaseTransform):
    type: Literal["ColorJitter"]
    brightness: Optional[Union[float, List[float]]] = None
    contrast: Optional[Union[float, List[float]]] = None
    saturation: Optional[Union[float, List[float]]] = None
    hue: Optional[Union[float, List[float]]] = None

# === Normalize ===
class NormalizeTransform(BaseTransform):
    type: Literal["Normalize"]
    mean: List[float]
    std: List[float]
    inplace: bool = False

# === ToTensor ===
class ToTensorTransform(BaseTransform):
    type: Literal["ToTensor"]

# === Sequential (Compose 역할) ===
class SequentialTransform(BaseTransform):
    type: Literal["Sequential"]
    transforms: List[BaseTransform]

# === 통합 TransformUnion ===
TransformUnion = Union[
    ResizeTransform,
    CenterCropTransform,
    RandomCropTransform,
    RandomHorizontalFlipTransform,
    RandomVerticalFlipTransform,
    RandomRotationTransform,
    ColorJitterTransform,
    NormalizeTransform,
    ToTensorTransform,
    SequentialTransform
]

class HyperParameters(BaseModel):
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 0.001
    #device_type: Optional[str] = None
    use_cloud: Optional[bool] = False

class ModelRequest(BaseModel):
    model_name: str
    layers: List[LayerUnion]
    dataset: str
    preprocessing: List[TransformUnion] = None
    hyperparameters: Optional[HyperParameters] = None

class CodeGenResponse(BaseModel):
    user_id: int
    code: str
    form: HyperParameters