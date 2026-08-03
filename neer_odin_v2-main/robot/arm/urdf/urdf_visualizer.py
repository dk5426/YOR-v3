import os
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

def main():
    # Get the directory of the URDF file
    urdf_path = os.path.join(os.path.dirname(__file__), "piper_description_right.xml")
    package_dir = os.path.join(os.path.dirname(urdf_path), "assets")

    # Load the URDF model
    model = pin.buildModelFromUrdf(urdf_path)

    # Create data required by the algorithms
    data = model.createData()

    # Create collision model and visual model with mesh directory
    collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION, package_dirs=[package_dir])
    visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL, package_dirs=[package_dir])

    # Create the visualizer with proper models
    viz = MeshcatVisualizer(model, collision_model, visual_model)

    # Initialize the viewer
    viz.initViewer()

    # Load the robot geometry and display it
    viz.loadViewerModel()

    # Display the robot in its initial configuration
    q = pin.neutral(model)
    viz.display(q)

    # Keep the window open
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
