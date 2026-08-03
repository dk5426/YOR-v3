from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pyarrow as pa


@dataclass
class BimanualPose:
    timestamp: int
    left_wxyz_xyz: npt.NDArray
    right_wxyz_xyz: npt.NDArray

    def encode(self) -> tuple[pa.Array, dict]:
        data = np.concatenate([
            self.left_wxyz_xyz.ravel(),
            self.right_wxyz_xyz.ravel(),
        ])
        return pa.array(data), {"timestamp": self.timestamp}

    @classmethod
    def decode(cls, data: pa.Array, metadata: dict) -> "BimanualPose":
        left_wxyz_xyz = (
            data[:7]
            .to_numpy()
            .reshape(
                7,
            )
        )
        right_wxyz_xyz = (
            data[7:]
            .to_numpy()
            .reshape(
                7,
            )
        )
        return cls(metadata["timestamp"], left_wxyz_xyz, right_wxyz_xyz)

@dataclass
class BimanualArmCommand:
    timestamp: int
    left_wxyz_xyz: npt.NDArray
    right_wxyz_xyz: npt.NDArray
    left_gripper: float
    right_gripper: float

    def encode(self) -> tuple[pa.Array, dict]:
        data = np.concatenate([
            self.left_wxyz_xyz.ravel(),
            self.right_wxyz_xyz.ravel(),
            np.array([self.left_gripper, self.right_gripper]),
        ])
        return pa.array(data), {"timestamp": self.timestamp}

    @classmethod
    def decode(cls, data: pa.Array, metadata: dict) -> "BimanualArmCommand":
        left_wxyz_xyz = (
            data[:7]
            .to_numpy()
            .reshape(7,))
        right_wxyz_xyz = (
            data[7:14]
            .to_numpy()
            .reshape(7,)
        )
        left_gripper = data[14].as_py()
        right_gripper = data[15].as_py()
        return cls(metadata["timestamp"], left_wxyz_xyz, right_wxyz_xyz, left_gripper, right_gripper)
