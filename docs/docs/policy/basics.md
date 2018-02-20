# RL Policy Basics

## Policies

See the `Policy` interface. The most important methods are `act`,
`update_from_episodes` and `update_from_replay_buffer`.

Note that all of these methods are also batched (i.e. they operate on multiple
episodes in parallel)

The model policy is the main one that we are trying to train. See
`MiniWoBPolicy` as an example.

## Configuration

## Model architecture

During training, there are several key systems involved:
- the environment
- policies
  - the model policy
  - the exploration policy
- the replay buffer


## Episodes



## Experience Buffer