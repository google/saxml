from typing import Any, Callable, Dict, List, Tuple

from typing import overload

class AdminOptions:
    timeout: float
    def __init__(self) -> None: ...
    def __copy__(self) -> AdminOptions: ...
    def __deepcopy__(self, arg0: dict) -> AdminOptions: ...

class AudioModel:
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    def Recognize(self, id: str, options: ModelOptions = ...) -> List[Tuple[str,float]]: ...
    @overload
    def Recognize(self, audio_bytes: str, options: ModelOptions = ...) -> List[Tuple[str,float]]: ...

class CustomModel:
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    def Custom(self, request: bytes, method_name: str, options: ModelOptions = ...) -> bytes: ...
    @overload
    def Custom(self, request: bytes, method_name: str, options: ModelOptions = ...) -> bytes: ...

class LanguageModel:
    def __init__(self, *args, **kwargs) -> None: ...
    def Embed(self, text: str, options: ModelOptions = ...) -> List[float]: ...
    def Generate(self, text: str, options: ModelOptions = ...) -> List[Tuple[str,float]]: ...
    def GenerateStream(self, text: str, callback: Callable, options: ModelOptions = ...) -> None: ...
    def Gradient(self, prefix: str, suffix: str, options: ModelOptions = ...) -> Tuple[List[float],Dict[str,List[float]]]: ...
    def Score(self, prefix: str, suffix: List[str], options: ModelOptions = ...) -> List[float]: ...

class Model:
    @overload
    def __init__(self, arg0: str, arg1: Options) -> None: ...
    @overload
    def __init__(self, arg0: str) -> None: ...
    def AM(self) -> AudioModel: ...
    def CM(self) -> CustomModel: ...
    def LM(self) -> LanguageModel: ...
    def MM(self) -> MultimodalModel: ...
    def VM(self) -> VisionModel: ...

class ModelDetail:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def active_replicas(self) -> int: ...
    @property
    def ckpt(self) -> str: ...
    @property
    def max_replicas(self) -> int: ...
    @property
    def model(self) -> str: ...
    @property
    def overrides(self) -> Dict[str,str]: ...

class ModelOptions:
    def __init__(self) -> None: ...
    def GetTimeout(self) -> float: ...
    def SetExtraInput(self, arg0: str, arg1: float) -> None: ...
    def SetExtraInputString(self, arg0: str, arg1: str) -> None: ...
    def SetExtraInputTensor(self, arg0: str, arg1: List[float]) -> None: ...
    def SetTimeout(self, arg0: float) -> None: ...
    def ToDebugString(self) -> str: ...
    def __copy__(self) -> ModelOptions: ...
    def __deepcopy__(self, arg0: dict) -> ModelOptions: ...

class ModelServerTypeStat:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def chip_topology(self) -> str: ...
    @property
    def chip_type(self) -> str: ...
    @property
    def num_replicas(self) -> int: ...

class MultimodalModel:
    def __init__(self, *args, **kwargs) -> None: ...
    def Generate(self, *args, **kwargs) -> Any: ...

class Options:
    fail_fast: bool
    num_conn: int
    proxy_addr: str
    def __init__(self) -> None: ...
    def __copy__(self) -> Options: ...
    def __deepcopy__(self, arg0: dict) -> Options: ...

class VisionModel:
    def __init__(self, *args, **kwargs) -> None: ...
    def Classify(self, image_bytes: str, options: ModelOptions = ...) -> List[Tuple[str,float]]: ...
    def Detect(self, image_bytes: str, text: List[str] = ..., options: ModelOptions = ...) -> List[Tuple[float,float,float,float,bytes,float]]: ...
    def Embed(self, image: str, options: ModelOptions = ...) -> List[float]: ...
    def ImageToImage(self, text: str, options: ModelOptions = ...) -> List[Tuple[bytes,float]]: ...
    def ImageToText(self, image_bytes: str, text: str = ..., options: ModelOptions = ...) -> List[Tuple[bytes,float]]: ...
    def TextAndImageToImage(self, text: str, image_bytes: str, options: ModelOptions = ...) -> List[Tuple[bytes,float]]: ...
    def TextToImage(self, text: str, options: ModelOptions = ...) -> List[Tuple[bytes,float]]: ...
    def VideoToText(self, image_frames: List[str], text: str = ..., options: ModelOptions = ...) -> List[Tuple[bytes,float]]: ...

def List(id: str, options: AdminOptions = ...) -> Tuple[str,str,int]: ...
def ListAll(id: str, options: AdminOptions = ...) -> List[str]: ...
def ListDetail(id: str, options: AdminOptions = ...) -> ModelDetail: ...
def Publish(id: str, model_path: str, checkpoint_path: str, num_replicas: int, options: AdminOptions = ...) -> None: ...
def StartDebugPort(arg0: int) -> None: ...
def Stats(id: str, options: AdminOptions = ...) -> List[ModelServerTypeStat]: ...
def Unpublish(id: str, options: AdminOptions = ...) -> None: ...
def Update(id: str, model_path: str, checkpoint_path: str, num_replicas: int, options: AdminOptions = ...) -> None: ...
def WaitForReady(id: str, num_replicas: int, options: AdminOptions = ...) -> None: ...