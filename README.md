# Web agents

**Authors**: Evan Zheran Liu\*, Kelvin Guu\*, Panupong (Ice) Pasupat\*, Tianlin Shi, Percy Liang (\* equal contribution)

Source code accompanying our ICLR 2018 paper: Reinforcement Learning on Web
Interfaces using Workflow-Guided Exploration.

Reproducible experiments using this code are located on [our Codalab
worksheet](https://worksheets.codalab.org/worksheets/0x0f25031bd42f4aabbc17625fe1484066/).


## Purpose

The goal of this project is to train machine learning models (agents) to do
things in a browser that can be specified in natural language, e.g. "Book a
flight from San Francisco to New York for Dec 23rd."

## Setup

### General setup

- Python dependencies
  ```
  pip install -r requirements.txt
  ```
  - If this gives you problems, try again and add pip's ```--ignore-installed```
  flag.

- Git submodules (needed for FormWoB and demonstrations)
  ```
  git submodule update --init --recursive
  ```

- Node and npm
  - Make sure Node and npm are installed via ```brew install node```. If they 
  are, ```node -v``` and ```npm -v``` should print version numbers.

- Torch
  - Install by following directions from http://pytorch.org/. Choose your OS,
  package manager, python version, and CUDA and you'll be given shell commands
  to run.

- Selenium
  - Outside this repository, download [ChromeDriver](https://sites.google.com/a/
    chromium.org/chromedriver/downloads). Unzip it and then add the directory
    containing the `chromedriver` executable to the `PATH` environment variable
    ```
    export PATH=$PATH:/path/to/chromedriver
    ```
  - If instead you're using Anaconda,
    use ```conda install -c conda-forge selenium```.

### Data directory setup

- This code depends on the environmental variable ```$RL_DATA``` being set,
  pointing to a configured data directory.

- Create a data directory ```mkdir -p /path/to/data``` and set ```export
  $RL_DATA=/path/to/data```. In order for the code to run, ```$RL_DATA```
  will need to be set to point at this directory.

- Next, set up the data directory:
  ```
  cd $RL_DATA
  # Download glove from https://nlp.stanford.edu/data/glove.6B.zip and place
  # in current directory however you want
  # Suggested: wget https://nlp.stanford.edu/data/glove.6B.zip
  unzip glove.6B.zip
  mv glove.6B glove
  ```

### Demonstration directory setup

```
# Where $REPO_DIR is the path to the root of this Git repository.
git clone https://github.com/stanfordnlp/miniwob-demos.git $REPO_DIR/third-party/miniwob-demos
export RL_DEMO_DIR=$REPO_DIR/third-party/miniwob-demos/
```

### MiniWoB setup

- There are 2 ways to access MiniWoB tasks:
  1. **Use the `file://` protocol (Recommended):**
    Open `miniwob-sandbox/html/` in the browser,
    and then export the URL to the `MINIWOB_BASE_URL` environment variable:
    ```
    export MINIWOB_BASE_URL='file:///path/to/miniwob-sandbox/html/'
    ```
  2. **Run a simple server:** go to `miniwob-sandbox/html/` and run the supplied `http-serve`.
    - The tasks should now be accessible at `http://localhost:8080/miniwob/`
    - To use a different port (say 8765), run `http-serve 8765`, and then
    export the following to the `MINIWOB_BASE_URL` environment variable:
    ```
    export MINIWOB_BASE_URL='http://localhost:8765/'
    ```
- Once you've followed one of the steps above, test `MiniWoBEnvironment` by running
  ```
  pytest wge/tests/miniwob/test_environment.py -s
  ```

### MiniWoB versions of FormWoB

Follow the "Run a simple server" instruction in the MiniWoB setup section above.

## Launching an Experiment

To train a model on a task, run:
```
python main.py configs/default-base.txt --task click-tab-2
```
- This executes the main entrypoint script, `main.py`. In particular, we pass it a base HOCON format config file and the task click-tab-2.
- Additional configs can be merged in by passing them as commandline arguments
  from configs/config-mixins
- Make sure that the following environmental variables are set:
  `MINIWOB_BASE_URL`, `RL_DEMO_DIR`, `REPO_DIR`.
- You may also want to set the `PYTHONPATH` to the same place as `REPO_DIR` to
  make imports work out properly
- You can also run this via docker by first running `python run_docker.py` to
  launch Docker and then running the above command. Unfortunately, you will
not be able to see the model train in the Docker container.
- The different tasks can be found in the subdirectories of
  third-party/miniwob-sandbox/html

If the script is working, you should see several Chrome windows pop up 
(operated by Selenium) and a training progress bar in the terminal.

## Experiment management

All training runs are managed by the `MiniWoBTrainingRuns` object. For example,
to get training run #141, do this:
```python
runs = MiniWoBTrainingRuns()
run = runs[141]  # a MiniWoBTrainingRun object
```

A `TrainingRun` is responsible for constructing a model, training it, saving it
and reloading it (see superclasses `gtd.ml.TrainingRun` and
`gtd.ml.TorchTrainingRun` for details.)

The most important methods on `MiniWobTrainingRun` are:
- `__init__`: the policy, the environment, demonstrations, etc, are all loaded
here.
- `train`: actual training of the policy happens here

## Model architecture

During training, there are several key systems involved:
- the environment
- policies
  - the model policy
  - the exploration policy
- episode generators
  - basic episode generator
  - best first episode generator
- the replay buffer

### Environment

All environments implement the `Environment` interface. A policy interacts
with the environment by calling the environment's `step` method and passing in
actions.

Note that an environment object is _batched_. It actually represents a batch
of environments, each running in parallel (so that we can train faster).

We mostly use `MiniWoBEnvironment` and `FormWoBEnvironment`.

### Policies

See the `Policy` interface. The most important methods are `act`,
`update_from_episodes` and `update_from_replay_buffer`.

Note that all of these methods are also batched (i.e. they operate on multiple
episodes in parallel)

The model policy is the main one that we are trying to train. See
`MiniWoBPolicy` as an example.

### Episode generators

See the `EpisodeGenerator` interface. An `EpisodeGenerator` runs a
`Policy` on an `Environment` to produce an `Episode`.

### Replay buffer

See the `ReplayBuffer` interface. A `ReplayBuffer` stores episodes produced
by the exploration policy. The final model policy is trained off episodes
sampled from the replay buffer.

## Configuration

All configs are in the `configs` folder. They are specified in HOCON format.
The arguments to `main.py` should be a list of paths to config files.
`main.py` then merges these config files according to the
[rules explained here](https://github.com/typesafehub/config/blob/master/HOCON.md#include-semantics-merging).
