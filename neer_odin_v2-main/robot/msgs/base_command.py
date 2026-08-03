from enum import IntEnum
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pyarrow as pa

class CommandType(IntEnum):
    BASE_VELOCITY = 1
    BASE_POSITION = 2
    LIFT_POSITION = 3

@dataclass
class BaseCommand:
    timestamp: int
    type: CommandType
    target: npt.NDArray

    def encode(self) -> tuple[pa.Array, dict]:
        data = np.concatenate([
            np.array([self.type.value]),
            self.target.ravel(),
        ])
        return pa.array(data), {"timestamp": self.timestamp}

    @classmethod
    def decode(cls, data: pa.Array, metadata: dict) -> "BaseCommand":
        command_type = CommandType(data[0].as_py())
        target = data[1:].to_numpy()
        return cls(metadata["timestamp"], command_type, target)



