from typing import Any, Dict
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

################################################################################################
#                                        Target config                                         #
################################################################################################
# features=tfds.features.FeaturesDict({
#     'steps': tfds.features.Dataset({
#         'observation': tfds.features.FeaturesDict({
#             'image': tfds.features.Image(
#                 shape=(128, 128, 3),
#                 dtype=np.uint8,
#                 encoding_format='jpeg',
#                 doc='Main camera RGB observation.',
#             ),
#         }),
#         'action': tfds.features.Tensor(
#             shape=(8,),
#             dtype=np.float32,
#             doc='Robot action, consists of [3x EEF position, '
#                 '3x EEF orientation yaw/pitch/roll, 1x gripper open/close position, '
#                 '1x terminate episode].',
#         ),
#         'discount': tfds.features.Scalar(
#             dtype=np.float32,
#             doc='Discount if provided, default to 1.'
#         ),
#         'reward': tfds.features.Scalar(
#             dtype=np.float32,
#             doc='Reward if provided, 1 on final step for demos.'
#         ),
#         'is_first': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on first step of the episode.'
#         ),
#         'is_last': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on last step of the episode.'
#         ),
#         'is_terminal': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on last step of the episode if it is a terminal step, True for demos.'
#         ),
#         'language_instruction': tfds.features.Text(
#             doc='Language Instruction.'
#         ),
#         'language_embedding': tfds.features.Tensor(
#             shape=(512,),
#             dtype=np.float32,
#             doc='Kona language embedding. '
#                 'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
#         ),
#     })
################################################################################################
#                                                                                              #
################################################################################################


def transform_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Maps step from source dataset to target dataset config.
       Input is dict of numpy arrays."""
    img = Image.fromarray(step['observation']['image']).resize(
        (128, 128), Image.Resampling.LANCZOS)
    curr_pos = step['observation']['state'][14:17]
    curr_quat = step['observation']['state'][17:21]
    curr_rot = R.from_quat(curr_quat)

    resi_pos = step['action'][:3]
    resi_axisAngle = step['action'][3:6]

    target_position = curr_pos + resi_pos
    target_position = np.array(target_position, dtype=np.float32)

    resi_rot = R.from_rotvec(resi_axisAngle)
    target_rot = resi_rot*curr_rot

    target_euler_angles = target_rot.as_euler('ZYX') # extrinsic 
    target_euler_angles = np.array(target_euler_angles, dtype=np.float32)

    # rotation angle verification
    # mat = target_rot.as_matrix()
    # yaw=np.arctan2(mat[1,0],mat[0,0])
    # pitch=np.arctan2(-mat[2,0],np.sqrt(mat[2,1]**2+mat[2,2]**2))
    # roll=np.arctan2(mat[2,1],mat[2,2])
    # print([yaw, pitch, roll], target_euler_angles)

    transformed_step = {
        'observation': {
            'image': np.array(img),
        },
        'action': np.concatenate(
            [target_position, target_euler_angles, np.array([0., int(step['is_last'])], dtype=np.float32) ]),
    }

    # copy over all other fields unchanged
    for copy_key in ['discount', 'reward', 'is_first', 'is_last', 'is_terminal',
                     'language_instruction', 'language_embedding']:
        transformed_step[copy_key] = step[copy_key]

    return transformed_step

