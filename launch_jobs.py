import argparse
import time
from os.path import join

from fabric.api import local

from script_tools import bash_string, upload_code, create_worksheet, task_lists, \
    upload_demos

# Parse arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-t', '--tasks')  # e.g. 'miniwob-hard', see script_tools.py:task_lists for more.
arg_parser.add_argument('-r', '--seed', type=int)   # random seed to run with
arg_parser.add_argument('-s', '--train_strategy')  # 'pge' or 'bc_rl'
arg_parser.add_argument('-n', '--num_demos', type=int)  # number of demos to use, between 0 and 32
arg_parser.add_argument('-c', '--up_code', action='store_true')  # whether to upload code
arg_parser.add_argument('-d', '--up_demos', action='store_true')  # whether to upload demos
args = arg_parser.parse_args()

# code is uploaded to the `web-agents` worksheet
# demonstrations are uploaded to the `web-agents` worksheet

# run bundles are uploaded to the corresponding task+strategy worksheet
# e.g. a 'miniwob-hard' job using `pge` will go to `web-agents-miniwob-hard-pge`

# The `web-agents` worksheet is a home page that points to all the other worksheets.

# Different sets of tasks (such as `miniwob-hard`) are defined at script_tools.py:task_lists


config_path = lambda sub_path: join('configs', sub_path)


def launch_job(task, train_strategy, num_demos, worksheet, demo_set, no_demo_filter, seed):
    """Launch a job.
    
    Sleeps for 10 sec after launching.
    
    CONFIGS ARE MERGED IN THE FOLLOWING ORDER:
        - default-base.txt
        - vanilla-rl.txt
        - <task>.txt
    
    Args:
        task (str): e.g. "click-checkboxes"
        train_strategy (str): can be one of the following:
            - bc_rl: behavior cloning plus RL
            - pge: program-guided exploration
        num_demos (int): number of demonstrations to BC on
        worksheet (str): name of target worksheet
        demo_set (str): name of the demo collection
        no_demo_filter (bool): if True, don't filter demos by reward
    """
    assert train_strategy in ('bc_rl', 'pge')

    docker_cmd_args = [
        'python main.py',
        '-t {}'.format(task),
        '-s "demonstrations.max_to_use = {}"'.format(num_demos),
        '-s "demonstrations.base_dir = {}"'.format(demo_set),
        '-r {}'.format(seed),
        config_path('default-base.txt'),
    ]

    if train_strategy == 'bc_rl':
        docker_cmd_args.append(config_path('config-mixins/bc-rl.txt'))

    if no_demo_filter:
        docker_cmd_args.append(config_path('config-mixins/no-demo-filter.txt'))

    docker_cmd = ' '.join(docker_cmd_args)

    launch_cmd_args = [
        'python run_codalab.py',
        '-w {}'.format(worksheet),
        '-n {}_{}'.format(task, train_strategy),
        bash_string(docker_cmd),
    ]
    launch_cmd = ' \\\n'.join(launch_cmd_args)
    local(launch_cmd)
    time.sleep(10)  # wait a bit between launches


# upload the latest code and demos
if args.up_demos:
    upload_demos('web-agents')
if args.up_code:
    upload_code('web-agents')

# get tasks
tasks = task_lists[args.tasks]

# select the right set of demos
if args.tasks == 'few-shot':
    if args.train_strategy == 'bc_rl':
        demo_set = '2017-10-26_third-turk'
    elif args.train_strategy == 'pge':
        demo_set = 'clean-demos'
    else:
        raise ValueError(args.tasks)
else:
    demo_set = '2017-10-16_second-turk'

# decide whether to filter demos
no_demo_filter = (args.tasks == 'few-shot')

# set worksheet
worksheet = 'web-agents-{}-{}-{}'.format(args.tasks, args.train_strategy, args.seed)
create_worksheet(worksheet)

for task in tasks:
    launch_job(task, args.train_strategy, args.num_demos, worksheet,
               demo_set, no_demo_filter, args.seed)
