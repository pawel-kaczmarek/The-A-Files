from abc import abstractmethod, ABC
from typing import ClassVar, List

import numpy as np

from taf.models.card import MethodCard


class SteganographyMethod(ABC):

    #: What the method is, for the catalogue, the UI and the documentation
    #: (``taf.models.card``). Packaged methods must declare one.
    card: ClassVar[MethodCard | None] = None

    @abstractmethod
    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        ...

    @abstractmethod
    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        ...

    @abstractmethod
    def type(self) -> str:
        ...
