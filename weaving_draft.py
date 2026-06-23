from typing import NamedTuple, Self
import re
import numpy as np
from numpy.typing import NDArray


class WeavingDraft(NamedTuple):
    """A draft for manual loom weaving
    - parsing from string represenatnion
    - compiling into numpy arrays of 1/0
    """

    tieup: NDArray[np.intc]  # shafts × treadles
    threading: NDArray[np.intc]  # shafts × heddles
    treadling: NDArray[np.intc]  # picks × treadles

    @property
    def shafts(self):
        return self.tieup.shape[0]

    @property
    def treadles(self):
        return self.tieup.shape[1]

    @property
    def dim(self):
        return self.threading.shape[1], self.treadling.shape[0]

    @staticmethod
    def parsetable(scheme: str):
        scheme = re.sub(r"\.", "-", re.sub(r"[^.\t]", "+", re.sub(r"\s+", "\t", scheme.strip())))
        rows = scheme.split("\t")
        return np.array([[c == "+" for c in row] for row in rows], dtype=np.intc)

    @classmethod
    def parse(cls, tieup: str, threading: str, treadling: str) -> Self:
        """Parse ascii scheme with '.' for empty and any marks"""
        return cls(cls.parsetable(tieup), cls.parsetable(threading), cls.parsetable(treadling))

    def compile(self) -> NDArray:
        """Compile resulting pattern of warp/weft cells as array of 1/0"""
        return self.treadling @ self.tieup.T @ self.threading
