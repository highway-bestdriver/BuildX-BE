from pydantic import BaseModel
from typing import List, Optional, Dict

class Layer(BaseModel):
    type: str
    id: str  # 각 레이어 고유 ID (필수)
    input: Optional[str] = None # 각 레이어 input 값
    filters: Optional[int] = None
    kernel_size: Optional[int] = None
    activation: Optional[str] = None
    pool_size: Optional[int] = None
    units: Optional[int] = None

class Preprocessing(BaseModel):
    resize: Optional[List[int]] = None
    normalize: Optional[bool] = False
    augmentation: Optional[bool] = False

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
