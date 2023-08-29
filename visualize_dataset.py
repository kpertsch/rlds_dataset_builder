import argparse
import tqdm
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb
import matplotlib
from einops import rearrange
matplotlib.use('Agg')


WANDB_ENTITY = None
WANDB_PROJECT = 'vis_rlds'


parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='name of the dataset to visualize')
args = parser.parse_args()

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY,
               project=WANDB_PROJECT)
else:
    render_wandb = False


# create TF dataset
dataset_name = args.dataset_name
print(f"Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
# ds = tfds.load(dataset_name, split='train',)
builder = tfds.builder_from_directory('/input/genom/imsquared_nonprehensile-1/1.0.0')
ds =builder.as_dataset(split='train[75%:]')

def tile_images(x: np.ndarray):
    """
    Tile input images to a nearest square grid.
    """
    assert (len(x.shape) == 4)  # NHWC
    n: int = len(x)
    dim = int(np.ceil(np.sqrt(n)))
    shape = (dim * dim,) + tuple(x[0].shape)

    # Create output buffer as a grid.
    grid = np.zeros(shape, dtype=x.dtype)

    # Copy input to grid.
    grid[:x.shape[0]] = x

    # Apply tiling.
    grid = rearrange(grid, '(d1 d2) h w c -> (d1 h) (d2 w) c',
                     d1=dim, d2=dim)
    return grid


ds = ds.shuffle(100)

# visualize episodes
for i, episode in enumerate(ds.take(5)):
    images = []
    for step in episode['steps']:
        images.append(step['observation']['image'].numpy())
    # image_strip = np.concatenate(images[::4], axis=1)
    image_strip = tile_images(np.stack(images[::4], axis=0))
    caption = step['language_instruction'].numpy().decode() + ' (temp. downsampled 4x)'
    print(caption)

    if render_wandb:
        wandb.log({f'image_{i}': wandb.Image(image_strip, caption=caption)})
    else:
        plt.figure()
        plt.imshow(image_strip)
        plt.title(caption)
    plt.savefig(F'/tmp/docker/image-{i:02d}.png')

# visualize action and state statistics
actions, states = [], []
for episode in tqdm.tqdm(ds.take(500)):
    for step in episode['steps']:
        actions.append(step['action'].numpy())
        states.append(step['observation']['state'].numpy())
actions = np.array(actions)
states = np.array(states)
action_mean = actions.mean(0)
state_mean = states.mean(0)

def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(5*n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem+1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(vector_mean[elem])
    plt.savefig(F'/tmp/docker/dataset-{tag}.png')

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})

vis_stats(actions, action_mean, 'action_stats')
vis_stats(states, state_mean, 'state_stats')

# if not render_wandb:
#     plt.show()

