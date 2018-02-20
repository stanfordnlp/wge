import argparse
import os
import random
import socket
from os.path import abspath, dirname, join
import numpy as np
import torch

from gtd.io import save_stdout
from gtd.log import set_log_level
from gtd.utils import Config
from wge.training_run import MiniWoBTrainingRuns


# CONFIGS ARE MERGED IN THE FOLLOWING ORDER:
# 1. configs in args.config_paths, from left to right
# 2. task config
# 3. config_strings


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-s', '--config_strings', action='append', default=[])
arg_parser.add_argument('-c', '--check_commit', default='strict')
arg_parser.add_argument('-p', '--profile', action='store_true')
arg_parser.add_argument('-d', '--description', default='None.')
arg_parser.add_argument('-n', '--name', default='unnamed')
arg_parser.add_argument('-r', '--seed', type=int, default=0)
arg_parser.add_argument('-t', '--task', required=True)
arg_parser.add_argument('config_paths', nargs='+')
args = arg_parser.parse_args()


set_log_level('WARNING')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# create run
runs = MiniWoBTrainingRuns(check_commit=(args.check_commit == 'strict'))

config_paths = args.config_paths
if len(config_paths) == 1 and config_paths[0].isdigit():
    # reload old run
    run = runs[int(config_paths[0])]
else:
    # new run according to configs
    configs = [Config.from_file(p) for p in config_paths]

    # add task config
    repo_dir = abspath(dirname(__file__))
    config_dir = join(repo_dir, 'configs')

    task_config_path = join(config_dir, 'task-mixins', '{}.txt'.format(args.task))
    if os.path.exists(task_config_path):
        # use existing config if it exists
        task_config = Config.from_file(task_config_path)
    else:
        # otherwise, create a very basic config
        task_config = Config.from_str('env.subdomain = {}'.format(args.task))
    configs.append(task_config)

    # add string configs
    configs.extend([Config.from_str(cfg_str) for cfg_str in args.config_strings])

    # validate all configs
    reference_config = Config.from_file(join(config_dir, 'default-base.txt'))
    for config in configs:
        config.validate(reference_config)

    # merge all configs together
    config = Config.merge(configs)  # later configs overwrite earlier configs
    run = runs.new(config, name=args.name)  # new run from config

    run.metadata['description'] = args.description
    run.metadata['name'] = args.name

run.metadata['host'] = socket.gethostname()

# start training
run.workspace.add_file('stdout', 'stdout.txt')
run.workspace.add_file('stderr', 'stderr.txt')

if args.profile:
    from gtd.chrono import Profiling, Profiler
    profiler = Profiler.default()
    import wge.wob_policy
    profiler.add_module(wge.wob_policy)
    import wge.miniwob.environment
    profiler.add_module(wge.miniwob.environment)
    import wge.training_run
    profiler.add_module(wge.training_run)
    import wge.episode_generator
    profiler.add_module(wge.episode_generator)
    import wge.miniwob.program_policy
    profiler.add_module(wge.miniwob.program_policy)
    import wge.miniwob.program
    profiler.add_module(wge.miniwob.program)
    Profiling.start()

with save_stdout(run.workspace.root):
    try:
        run.train()
    finally:
        run.close()
        if args.profile:
            Profiling.report()
