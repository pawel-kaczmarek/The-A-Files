from abc import ABC, abstractmethod
from numbers import Number

import numpy as np


class Metric(ABC):

    #: ``True`` when a higher score means the processed signal is closer to the
    #: original, ``False`` when a lower one does, ``None`` when the score has
    #: no such direction. Rankings and significance tests need the direction;
    #: guessing it from the metric's name is how distances ended up ranked as
    #: if larger were better.
    higher_is_better: bool | None = None

    #: Names of the entries when ``calculate`` returns several numbers, in
    #: order. Each entry is reported on its own; averaging them would mix, for
    #: example, a signal-to-distortion ratio with a permutation index.
    components: tuple[str, ...] = ()

    #: Direction of each component, aligned with ``components``. Empty means
    #: every component follows ``higher_is_better``; ``None`` marks an entry
    #: that is reported but never ranked (a cover-only reference score).
    component_directions: tuple[bool | None, ...] = ()

    @abstractmethod
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    def labelled_values(self, value: Number | np.ndarray) -> dict[str, float]:
        """The result of ``calculate`` as named scalar values.

        A scalar keeps the metric's name. An array is split into one entry per
        component, ``"<name> [<component>]"``; entries of an array without
        declared components are numbered rather than averaged.
        """
        array = np.asarray(value, dtype=np.float64).ravel()
        if array.size == 1:
            return {self.name(): float(array[0])}
        labels = self.components if len(self.components) == array.size else tuple(
            str(index) for index in range(array.size)
        )
        return {f"{self.name()} [{label}]": float(entry) for label, entry in zip(labels, array)}

    def directions(self) -> dict[str, bool | None]:
        """Direction of every label ``labelled_values`` can produce."""
        if not self.components:
            return {self.name(): self.higher_is_better}
        directions = self.component_directions or (self.higher_is_better,) * len(self.components)
        return {
            f"{self.name()} [{label}]": direction
            for label, direction in zip(self.components, directions)
        }
