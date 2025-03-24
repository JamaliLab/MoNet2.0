# MoNet2.0
Dilated CNN classifier for anomlous diffusion models

## Installation
-First create a conda enviornment for MoNet2.0 with python 3.10.14: `conda create -n monet2 python=3.10.14`

-Then install the requirements: `pip install -r requirements.txt`

## Usage
-`run_monet2_classification.py` runs the trained MoNet2.0 to generate the confusion matrix for the model showing the model classification performance, and a bar chart showing the percentage of LEONARDO-Generated trajectories classified as different diffusion classes.

-`run_monet2_fd.py` runs the trained MoNet2.0 to generate a lower triangular matrix of Frechet distance values. Each cell in the matrix represents the Frechet distance between the second-last layer of MoNet2.0 with input trajectories from different diffusion classes.

-A trained MoNet2.0 model is present in the repository as `monet2.keras`. To train MoNet2.0 from scratch, use the `monet_train.py` file. Download the training dataset from Huggingface: [https://huggingface.co/datasets/JamaliLab/MoNet2.0](https://huggingface.co/datasets/JamaliLab/MoNet2.0)

For generating and analyzing trajectories using LEONARDO, check out [https://github.com/JamaliLab/LEONARDO](https://github.com/JamaliLab/LEONARDO).
