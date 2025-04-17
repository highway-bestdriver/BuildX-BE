from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Literal

class Layer(BaseModel):
    type: str
    id: str
    input: Optional[str] = None

    # Conv2D, DepthwiseConv2D
    filters: Optional[int] = None
    kernel_size: Optional[Union[int, List[int]]] = None
    strides: Optional[Union[int, List[int]]] = None
    padding: Optional[str] = None
    activation: Optional[str] = None
    dilation_rate: Optional[Union[int, List[int]]] = None
    use_bias: Optional[bool] = None
    depth_multiplier: Optional[int] = None

    # Pooling
    pooling_type: Optional[str] = None
    pooling_size: Optional[Union[int, List[int]]] = None

    # Dense
    units: Optional[int] = None

    # Dropout
    rate: Optional[float] = None
    seed: Optional[int] = None

    # BatchNorm
    axis: Optional[int] = None
    momentum: Optional[float] = None
    epsilon: Optional[float] = None

    # Upsampling, Concatenate, Add
    size: Optional[List[int]] = None
    residual_connection: Optional[bool] = None

class Preprocessing(BaseModel):
    resize: Optional[List[int]] = None  # [height, width]
    normalize: Optional[bool] = False
    augmentation: Optional[bool] = False

    random_crop_height: Optional[int] = None
    random_crop_width: Optional[int] = None
    random_crop_seed: Optional[int] = None

    random_contrast_factor: Optional[float] = None  # [0,1]
    random_contrast_seed: Optional[int] = None

    random_flip_mode: Optional[Literal["horizontal", "vertical", "horizontal_and_vertical"]] = None
    random_flip_seed: Optional[int] = None

    random_rotation_factor: Optional[float] = None  # [0,1]
    random_rotation_seed: Optional[int] = None

    random_translation_height_factor: Optional[float] = None  # [0,1]
    random_translation_width_factor: Optional[float] = None  # [0,1]
    random_translation_seed: Optional[int] = None

class HyperParameters(BaseModel):
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 0.001
    device_type: Optional[str] = None

class ModelRequest(BaseModel):
    model_name: str
    layers: List[Layer]
    dataset: str
    preprocessing: Optional[Preprocessing] = None
    hyperparameters: Optional[HyperParameters] = None

class CodeGenResponse(BaseModel):
    user_id: int
    code: str
    form: HyperParameters
