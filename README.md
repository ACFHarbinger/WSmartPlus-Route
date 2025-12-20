# WSmart+ Route
Machine Learning models and Operations Research solvers for Combinatorial Optimization problems, focusing on route planning for waste collection.

## Tech Stack
- [Python Programming Language](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Gurobi Optimizer](https://www.gurobi.com/)
- [Hexaly Optimizer](https://www.hexaly.com/)

### CUDA Drivers
You need to have the CUDA drivers installed in order to be able to run the program on NVidia GPU. If you need to install the drivers, you can [download them here](https://developer.nvidia.com/cuda-downloads), and then follow (the instructions on this website)[https://docs.nvidia.com/cuda/index.html] to install them on your operating system.

### Adapted Code
This project contains code or ideas that were adapted from the following repositories:
- [Attention, Learn to Solve Routing Problems](https://github.com/wouterkool/attention-learn-to-route)
- [Heterogeneous Attentions for Solving PDP via DRL](https://github.com/jingwenli0312/Heterogeneous-Attentions-PDP-DRL)
- [POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://github.com/yd-kwon/POMO/tree/master)
- [WSmart+ Bin Analysis](https://github.com/ACFPeacekeeper/wsmart_bin_analysis)
- [Do We Need Anisotropic Graph Neural Networks?](https://github.com/shyam196/egc)
- [Learning TSP Requires Rethinking Generalization](https://github.com/chaitjo/learning-tsp)
- [HGS-CVRP: A modern implementation of the Hybrid Genetic Search for the CVRP](https://github.com/vidalt/HGS-CVRP)

This repository also includes adaptions of the following repositories as baselines:
* https://github.com/MichelDeudon/encode-attend-navigate
* https://github.com/mc-ride/orienteering
* https://github.com/jordanamecler/PCTSP
* https://github.com/rafael2reis/salesman

## Setup Dependencies
You can choose to install this repository's dependencies using any of the following methods below.

### UV
To use the [UV Python package and project manager](https://github.com/astral-sh/uv) to setup the virtual environment, you just have to synchronize the project.
```bash
uv sync
```

Afterwards, you can initialize the virtual environment by running one of the following commands: 
- On the Linux CLI: `source .venv/bin/activate`
- On the Windows CMD: `.venv\Scripts\activate.bat`
- On the Windows PS: `.venv\Scripts\Activate.ps1`

After activating the virtual environment, you can list the installed packages in a similar manner to Conda by using Pip through UV:
```bash
uv pip list
```

Also, if you want to deactivate and/or delete the created virtual environment you can execute the following command(s).
```bash
deactivate
rm -rf .venv
```

#### UV Installation
To install UV, you simply need to execute the command `curl -LsSf https://astral.sh/uv/install.sh | sh` on the Linux CLI (or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on the Windows CMD|PS).

### Anaconda Environment
To setup the environment for the project using the [Anaconda distribution](https://www.anaconda.com/), you just need to run the following commands in the main directory:
```bash
conda env create --file env/environment.yml -y --name wsr
conda activate wsr
```

To list the installed packages (and their respective versions), just run the following command after activating the Conda environment:
```bash
conda list
```

and if you want to deactivate and/or delete the previously created Conda environment:
```bash
conda deactivate
conda remove -n wsr --all -y
```

#### Conda Installation
If you need to install conda beforehand, you just need to run the following commands (while replacing the variables for the values you want to use, which determine your Anaconda version):
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-<year>.<month>-<version_id>-Linux-x86_64.sh
bash Anaconda3-<year>.<month>-<version_id>-Linux-x86_64.sh
```
For this project, we recommend you use Anaconda 3 with year=2024, month=10, version_id=1.

### Virtual Environment
To setup the virtual environment for the project using the Pip package installer and Python's venv module:
```bash
python3 -m venv env/.wsr
source env/.wsr/bin/activate
pip install -r env/requirements.txt
pip install -r env/pip_requirements.txt
```

After activating the virtual environment, you can list the installed packages in a similar manner to Conda by using Pip:
```bash
pip list
```

and if you want to deactivate and/or delete the created virtual environment:
```bash
deactivate
rm -rf env/.wsr
```

Note: to use this method, you already need to have the correct version of Python 3 already installed in your system.

### Setup Scripts
You can also execute a script to completely setup your virtual environment using your preferred method. To do that, you simply need to execute the Linux command
```bash
bash scripts/setup_env.sh <selected_method>
```

or the following command on the Windows CMD:
```cmd
scripts\setup_env.bat <selected_method>
```

Note: the selected_method variable shoud be replaced with -> uv|conda|venv

### Setup Optimizers
#### Gurobi
To use the Gurobi optimization software, you first need to [login or create an account](https://portal.gurobi.com/iam/login/) on their website. Then, you can [request a license](https://portal.gurobi.com/iam/licenses/list) and/or [download the software](https://www.gurobi.com/downloads/).

#### Hexaly
To use the Hexaly optimizer, you simply need to [login or create an account](https://www.hexaly.com/login), and then [request a license](https://www.hexaly.com/account/on-premise-licenses) on their website.

## Program Usage
### Generating Data
Training data can be generated on the fly. To pre-generate training data for graphs with 50 vertices for 10 epoches (e.g., for the vehicle routing problems with profits - 'vrpp'):
```bash
python main.py generate_data virtual --problem vrpp --graph_sizes 50 --n_epochs 10 --seed 42 --data_distribution gamma1
```

To generate validation and test data for graphs with 20 and 50 vertices (e.g., for all problems):
```bash
python main.py generate_data val --problem all --graph_sizes 20 50 --seed 42 --data_distribution gamma1
python main.py generate_data test --problem all --graph_sizes 20 50 --seed 42 --data_distribution gamma1
```

### Training
In order to train the Attention Model (AM) on vrpp instances with 50 vertices using rollout as the REINFORCE baseline and using the generated datasets:
```bash
python main.py train --graph_size 50 --baseline rollout --train_dataset virtual --val_dataset data/vrpp/vrpp20_val_seed1234 --data_distribution gamma1 --n_epochs 10
```

To train the Transformer-Graph Convolutional Network (TransGCN) model on vrpp instances with 20 vertices (with edges for the 4 nearest neighbors of each vertex) for 20 epochs using REINFORCE without a baseline and generating the training data on the fly:
```bash
python main.py train --model transgcn --graph_size 20 --edge_threshold 0.3 --edge_threshold 0.2 --edge_method "knn" --n_epochs 20 --data_distribution gamma1
```

#### Resume Training
In order to load a pretrained model and the optimizer state, and to resume the previous training session for an additional 5 epochs:
```bash
python main.py train --model transgcn --graph_size 20 --edge_threshold 0.2 --edge_method "knn" --n_epochs 5 --epoch_start 20 --load_path "results/vrpp_20/run_{datetime}/epoch-19.pt" --data_distribution gamma1
```

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py train 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 vertices).

### Evaluation
To evaluate a model (e.g., the AM), you can add the `--eval-only` flag to `main.py train`, or use `main.py evaluate`, which will additionally measure timing and save the results:
```bash
python main.py eval data/vrpp/vrpp20_test_seed1234.pkl --model assets/model_weights/vrpp_20/am --decode_strategy greedy --data_distribution gamma1
```
Note: If the epoch is not specified, by default the last one in the folder will be used.

#### Sampling
To report the best of 1280 sampled solutions, use the following command:
```bash
python main.py eval data/vrpp/vrpp20_test_seed1234.pkl --model assets/model_weights/vrpp_20/am --decode_strategy sample --width 1280 --eval_batch_size 1 --data_distribution gamma1
```
The Beam Search algorithm can be used with the flags `--decode_strategy bs --width {beam_size}`.

### Testing on Simulator
To test all the available policies (excluding the models) on the Gamma distribution and a road network with 20 bins for 31 days, using all available CPU cores:
```bash
python main.py test_sim --policies policy_last_minute policy_last_minute_and_path policy_regular policy_look_ahead_a policy_look_ahead_b gurobi --problem vrpp --size 20 --days 31 --data_distribution gamma1 --cf 50 70 90 --lvl 2 3 6 --gp 0.84 --n_vehicles 1
```

To test the TransGCN model on the Empirical distribution and a road network with 20 bins and on the 4 nearest neighbors for 365 days, using a single CPU core:
```bash
python main.py test_sim --policies transgcn --problem vrpp --size 20 --edge_threshold 0.2 --edge_method "knn" --days 365 --data_distribution emp --cpu_cores 1
```

#### Multiple Samples
To test all the available policies (excluding the models) for 10 samples on the Gamma distribution (option 2) and a road network with 100 bins for 365 days, using all available CPU cores:
```bash
python main.py test_sim --policies policy_last_minute policy_last_minute_and_path policy_regular policy_look_ahead_a policy_look_ahead_b gurobi --problem vrpp --size 100 --days 365 --data_distribution gamma2 --cf 50 70 90 --lvl 2 3 6 --gp 0.84 --n_vehicles 1 --n_samples 10
```

You can also resume a previously unfinished test run by selecting the policies with missing results for some (or all) of the samples:
```bash
python main.py test_sim --policies policy_look_ahead_a policy_look_ahead_b gurobi --problem vrpp --size 100 --days 365 --data_distribution gamma2 --gp 0.84 --n_vehicles 1 --n_samples 10 --resume
```

### Graphical User Interface (GUI)
You can execute the commands using a GUI. To activate the GUI, simply run the following command:
```bash
python main.py gui [--test_only]
```

### Scripts
There are [several scripts](/scripts/) included in this repository that allow you to run the program with all the arguments you require by simply executing `bash scripts/<script_name>.sh`. Currently, the functionalities that are available using scripts (i.e., the possible values for <script_name>) are:
- Generate datasets for training, validation, or testing: [gen_data](/scripts/gen_data.sh)
- Training Deep Learning models to perform Combinatorial Optimization problems: [train](/scripts/train.sh)
- Training Deep Learning models using hyper-parameter optimization: [hyperparam_optim](/scripts/hyperparam_optim.sh)
- Test the policies and previously trained models on the WSmart+ Route simulator: [test_sim](/scripts/test_sim.sh)
- Perform any of the previous functionalities on a Slurm server/cluster: [slurm](/scripts/slurm.sh) and [slim_slurm](/scripts/slim_slurm.sh)

You can run the equivalent scripts in Windows (except the slurms scripts) by executing the following command on the CMD:
```cmd
scripts\<script_name>.bat
```

### Test Suite
Additionally, you can run a test suite to verify if the program's various functionalities are being executed as intended:
```bash
python main.py test_suite [--module <module_name>|--class <class_name>|--test <test_name>|--markers <marker_name>..<marker_name>]
```

Note: running the test suite without any of the previous arguments results in performing all available tests for the program.

## Build Distribution
If you have UV, you can build the source and binary distribution for this project simply by running:
```bash
uv build
```

### Create Executable
Finally, you can also use the PyInstaller module to create an executable of the program by running the following command:
```bash
pyinstaller build.spec [--clean] [--noconsole]
```