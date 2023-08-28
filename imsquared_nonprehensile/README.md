TODO(example_dataset): Markdown description of your dataset.
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):


# IMsquared Non-Prehensile Manipulation Dataset

## Overview
The IMsquared Non-Prehensile Manipulation Dataset is a collection of scenarios involving various non-prehensile manipulation actions performed on different objects. The dataset encompasses pushing, toppling, sliding, and yaw twisting actions, each contributing to a comprehensive exploration of non-grasping-based interactions with objects. This dataset is designed to support research and development in robotics and computer vision, particularly in the field of object pose estimation and non-prehensile manipulation strategies.

## Contents
The dataset comprises the following key components:

- **Actions**: The dataset includes examples of the following non-prehensile manipulation actions:
  - Pushing: Objects are subjected to external forces to induce translational movement.
  - Toppling: Objects are caused to fall from an initial stable position to another orientation.
  - Sliding: Objects are made to glide across surfaces without lifting.
  - Yaw Twisting: Objects are rotated around the vertical axis.

- **Objects**: A variety of objects with diverse shapes, sizes, and materials are used in the dataset. This ensures a broad range of scenarios and challenges for pose estimation and manipulation.

- **Data Modalities**:
  - Partial Point Clouds: Partial point cloud data captures the 3D structure of the scene, enabling accurate object localization and pose estimation.
  - RGB Images: RGB images provide visual information to support object identification and pose estimation.
  - Robot Proprioceptive Data: Robot joint position and velocity data to provide current robot state.

