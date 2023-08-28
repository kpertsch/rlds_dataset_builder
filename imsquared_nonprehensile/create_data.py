import numpy as np
import pickle
import torch
import os
import numpy as np
import tqdm
import os

N_TRAIN_EPISODES = 2
# EPISODE_LENGTH = 100


def create_episode(episode_log_path, output_path):
    episode = []

    file_names = sorted([f for f in os.listdir(episode_log_path) if (f.endswith('.pkl') and not f.startswith('task'))], key=lambda x: int(x.split('.')[0]))
    instruction = 'toppling a jar' # TODO: instruction should be gathered from task.pkl

    robot_frame_offset = np.array([0.5, 0.0, -0.4], dtype=np.float32)

    for filename in file_names:
        file_path = os.path.join(episode_log_path, filename)
        print(f'{filename} is processing')
        with open(file_path, 'rb') as file:
            log = pickle.load(file)

            
            image = log.get('color') # (480, 640, 3)
            # depth = log.get('depth') # (480, 640)
            partial_pointcloud = log.get('partial_cloud')[0] # (512, 3)
            partial_pointcloud += robot_frame_offset
            robot_state = log.get('robot_state')[0] # (14)
            hand_state = log.get('hand_state').cpu().detach().numpy()[0] # (7)
            hand_state[:3] += robot_frame_offset
            state = np.concatenate((robot_state, hand_state)) # (21)
            action = log.get('action')[0] # (20)
            language_instruction = instruction

            episode.append({
                        'image': image,
                        'state': state,
                        # 'depth': depth,
                        'partial_pointcloud': partial_pointcloud,
                        'action': action,
                        'language_instruction': language_instruction,
                    })

    np.save(output_path, episode)


print("Generating train examples...")
os.makedirs('data/train', exist_ok=True)
episode_log_path = '/home/user/workspace/rlds_dataset_builder/train_data/log'
for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    create_episode(episode_log_path+f'_{i}', f'data/train/episode_{i}.npy')

print('Successfully created example data!')


# with open('/home/user/workspace/rlds_dataset_builder/data/log/task.pkl', 'rb') as f:
#     data=pickle.load(f)

