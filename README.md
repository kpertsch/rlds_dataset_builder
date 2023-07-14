# RLDS Dataset Conversion

This repo demonstrates how to convert an existing dataset into RLDS format for RT-X integration.
It provides an example for converting a dummy dataset to RLDS. To convert your own dataset, fork this repo and 
modify the example code for your dataset following the steps below.

## Installation

First create a conda environment using the provided environment.yml file:
```
conda env create -f environment.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `matplotlib`, `plotly` and `wandb`.


## Run Example RLDS Dataset Creation

Before modifying the code to convert your own dataset, run the provided example dataset creation script to ensure
everything is installed correctly. Run the following lines to create some dummy data and convert it to RLDS.
```
cd example_dataset
python3 create_example_data.py
tfds build
```

This should create a new dataset in `~/tensorflow_datasets/example_dataset`. Please verify that the example
conversion worked before moving on.


## Converting your Own Dataset to RLDS

Now we can modify the provided example to convert your own data. Follow the steps below:

1. **Rename Dataset**: Change the name of the dataset folder from `example_dataset` to the name of your dataset (e.g. robo_net_v2), 
also change the name of `example_dataset_dataset_builder.py` by replacing `example_dataset` with your dataset's name (e.g. robo_net_v2_dataset_builder.py)
and change the class name `ExampleDataset` in the same file to match your dataset's name, using camel case instead of underlines (e.g. RoboNetV2).

2. **Modify Features**: Modify the data fields you plan to store in the dataset. You can find them in the `_info()` method
of the `ExampleDataset` class. Please add **all** data fields your raw data contains, i.e. please add additional features for 
additional cameras, audio, tactile features etc. If your type of feature is not demonstrated in the example (e.g. audio),
you can find a list of all supported feature types [here](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?hl=en#classes).
You can store step-wise info like camera images, actions etc in `'steps'` and episode-wise info like `collector_id` in `episode_metadata`.
Please don't remove any of the existing features in the example (except for `wrist_image`), since they are required for RLDS compliance.
Note that we store `language_instruction` in every step even though it is episode-wide information for easier downstream usage.

3. **Modify Dataset Splits**: The function `_split_generator()` determines the splits of the generated dataset (e.g. training, validation etc.).
If your dataset defines a train vs validation split, please provide the corresponding information to `_generate_examples()`, e.g. 
by pointing to the corresponding folders (like in the example) or file IDs etc. If your dataset does not define splits,
remove the `val` split and only include the `train` split. You can then remove all arguments to `_generate_examples()`.

4. **Modify Dataset Conversion Code**: Next, modify the function `_generate_examples()`. Here, your own raw data should be 
loaded, filled into the episode steps and then yielded as a packaged example. Note that the value of the first return argument,
`episode_path` in the example, is only used as a sample ID in the dataset and can be set to any value that is connected to the 
particular stored episode, or any other random value. Just ensure to avoid using the same ID twice.

5. **Provide Dataset Description**: Next, add a bibtex citation for your dataset in `CITATIONS.bib` and add a short description
of your dataset in `README.md` inside the dataset folder. You can also provide a link to the dataset website.

That's it! You're all set to run dataset conversion. Inside the dataset directory, run:
```
tfds build
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/<name_of_your_dataset>`.

## Visualize Converted Dataset
To verify that the data is converted correctly, please run the data visualization script from the base directory:
```
python3 visualize_dataset.py <name_of_your_dataset>
``` 
This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.
Note, if you are running on a headless server you can modify `WANDB_ENTITY` at the top of `visualize_dataset.py` and 
add your own WandB entity -- then the script will log all visualizations to WandB. 