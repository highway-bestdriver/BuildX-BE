from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Literal

class BaseLayer(BaseModel):
    type: str
    id: str
    input: Optional[str] = None

class InputLayer(BaseLayer):
    type: Literal["Input"]
    shape: List[int]

# Conv2D
class Conv2DLayer(BaseLayer):
    type: Literal["Conv2D"]
    filters: int
    kernel_size: Union[int, List[int]]
    strides: Optional[Union[int, List[int]]] = [1, 1]
    padding: Optional[Literal["valid", "same"]] = "same"
    activation: Optional[str] = None
    dilation_rate: Optional[Union[int, List[int]]] = None
    use_bias: Optional[bool] = True

# DepthwiseConv2D
class DepthwiseConv2DLayer(BaseLayer):
    type: Literal["DepthwiseConv2D"]
    kernel_size: Union[int, List[int]]
    depth_multiplier: Optional[int] = 1
    activation: Optional[str] = None
    padding: Optional[str] = "same"

# Pooling
class PoolingLayer(BaseLayer):
    type: Literal["Pooling"]
    pooling_type: Literal["max", "avg"]
    pooling_size: Union[int, List[int]]
    strides: Optional[Union[int, List[int]]] = [2, 2]
    padding: Optional[Literal["valid", "same"]] = "same"

# Dense
class DenseLayer(BaseLayer):
    type: Literal["Dense"]
    units: int
    activation: Optional[str] = None
    use_bias: Optional[bool] = True

# Dropout
class DropoutLayer(BaseLayer):
    type: Literal["Dropout"]
    rate: float
    seed: Optional[int] = None

# BatchNormalization
class BatchNormLayer(BaseLayer):
    type: Literal["BatchNorm"]
    axis: Optional[int] = -1
    momentum: Optional[float] = 0.99
    epsilon: Optional[float] = 0.001

# Flatten
class FlattenLayer(BaseLayer):
    type: Literal["Flatten"]

# Upsampling
class UpsamplingLayer(BaseLayer):
    type: Literal["Upsampling"]
    size: List[int]

# Concatenate
class ConcatenateLayer(BaseLayer):
    type: Literal["Concatenate"]
    axis: Optional[int] = -1

# Add
class AddLayer(BaseLayer):
    type: Literal["Add"]
    id: str
    input: List[str]  # 2개 이상의 입력 받기
    residual_connection: Optional[bool] = False  # 필요 시 skip connection 표시용

# ReLU
class ReLULayer(BaseLayer):
    type: Literal["ReLU"]

# LeakyReLU
class LeakyReLULayer(BaseLayer):
    type: Literal["LeakyReLU"]
    alpha: Optional[float] = 0.3  # 기본값은 Keras 디폴트

# Sigmoid
class SigmoidLayer(BaseLayer):
    type: Literal["Sigmoid"]

# Tanh
class TanhLayer(BaseLayer):
    type: Literal["Tanh"]

# Softmax
class SoftmaxLayer(BaseLayer):
    type: Literal["Softmax"]
    axis: Optional[int] = -1  # Keras 기본값



LayerUnion = Union[
    InputLayer,
    Conv2DLayer,
    DepthwiseConv2DLayer,
    PoolingLayer,
    DenseLayer,
    DropoutLayer,
    BatchNormLayer,
    FlattenLayer,
    UpsamplingLayer,
    ConcatenateLayer,
    AddLayer,
    ReLULayer,
    LeakyReLULayer,
    SigmoidLayer,
    TanhLayer,
    SoftmaxLayer
]

# 전처리
class Resize(BaseModel):
    height: int
    width: int

class RandomCrop(BaseModel):
    height: int
    width: int
    seed: Optional[int] = None

class RandomContrast(BaseModel):
    factor: float  # [0,1]
    seed: Optional[int] = None

class RandomFlip(BaseModel):
    mode: Literal["horizontal", "vertical", "horizontal_and_vertical"]
    seed: Optional[int] = None

class RandomRotation(BaseModel):
    factor: float  # [0,1]
    seed: Optional[int] = None

class RandomTranslation(BaseModel):
    height_factor: float  # [0,1]
    width_factor: float   # [0,1]
    seed: Optional[int] = None

class Preprocessing(BaseModel):
    resize: Optional[Resize] = None
    normalize: Optional[bool] = None
    augmentation: Optional[bool] = False
    random_crop: Optional[RandomCrop] = None
    random_contrast: Optional[RandomContrast] = None
    random_flip: Optional[RandomFlip] = None
    random_rotation: Optional[RandomRotation] = None
    random_translation: Optional[RandomTranslation] = None

class HyperParameters(BaseModel):
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 0.001
    device_type: Optional[str] = None

class ModelRequest(BaseModel):
    model_name: str
    layers: List[LayerUnion]
    dataset: str
    preprocessing: Optional[Preprocessing] = None
    hyperparameters: Optional[HyperParameters] = None

class CodeGenResponse(BaseModel):
    user_id: int
    code: str
    form: HyperParameters
