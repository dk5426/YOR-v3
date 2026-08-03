from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pyarrow as pa

@dataclass
class Pose:
    timestamp: int
    wxyz_xyz: npt.NDArray

    def encode(self) -> tuple[pa.Array, dict]:
        return pa.array(self.wxyz_xyz.ravel()), {"timestamp": self.timestamp}

    @classmethod
    def decode(cls, data: pa.Array, metadata: dict) -> "Pose":
        return cls(metadata["timestamp"], data.to_numpy().reshape(7,))

@dataclass
class ArmCommand:
    timestamp: int
    wxyz_xyz: npt.NDArray
    gripper: float

    def encode(self) -> tuple[pa.Array, dict]:
        data = np.concatenate([
            self.wxyz_xyz.ravel(),
            np.array([self.gripper]),
        ])
        return pa.array(data), {"timestamp": self.timestamp}

    @classmethod
    def decode(cls, data: pa.Array, metadata: dict) -> "ArmCommand":
        wxyz_xyz = (
            data[:7]
            .to_numpy()
            .reshape(7,))
        gripper = data[7].as_py()
        return cls(metadata["timestamp"], wxyz_xyz, gripper)