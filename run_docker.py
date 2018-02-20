#!/u/nlp/packages/anaconda2/bin/python
import os
import argparse

from os.path import dirname, abspath
from fabric.api import local
from script_tools import bash_string, check_cwd_is_repo_root


check_cwd_is_repo_root()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-r', '--root', action='store_true', help='Run as root in Docker.')
arg_parser.add_argument('-g', '--gpu', default='', help='GPU to use.')
arg_parser.add_argument('-d', '--debug', action='store_true', help='Print command instead of running.')
arg_parser.add_argument('command', nargs='?', default=None,
                        help='Command to execute in Docker. If no command is specified, ' \
                             'you enter interactive mode. ' \
                             'To execute a command with spaces, wrap ' \
                             'the entire command in quotes.')
args = arg_parser.parse_args()

repo_dir = abspath(dirname(__file__))
image = 'kelvinguu/web-agents:1.0'  # docker image
my_uid = local('echo $UID', capture=True)
data_dir = os.environ['RL_DATA']

docker_args = [
               "--net host",  # access to the Internet
               "--publish 8888:8888",  # only certain ports are exposed
               "--publish 6006:6006",
               "--publish 8080:8080",
               "--ipc=host",
               "--rm",
               "--volume {}:/data".format(data_dir),
               "--volume {}:/code".format(repo_dir),
               "--env RL_DATA=/data",
               "--env PYTHONPATH=/code",
               "--env RL_DEMO_DIR=/code/third-party/miniwob-demos/",
               "--env MINIWOB_BASE_URL=file:///code/miniwob-sandbox/html/",
               "--env CUDA_VISIBLE_DEVICES={}".format(args.gpu),
               "--workdir /code"]

# interactive mode
if args.command is None:
    docker_args.append('--interactive')
    docker_args.append('--tty')
    args.command = '/bin/bash'

if not args.root:
    docker_args.append('--user={}'.format(my_uid))

if args.gpu == '':
    # run on CPU
    docker = 'docker'
else:
    # run on GPU
    docker = 'nvidia-docker'

pull_cmd = "docker pull {}".format(image)

# start a virtual screen in the background, then execute the specified command
screen_cmd = "Xvfb :99 -screen 0 1024x768x16 &> xvfb.log & export DISPLAY=:99.0"
exec_cmd = "({}) && {}".format(screen_cmd, args.command)

run_cmd = '{docker} run {options} {image} /bin/bash -c {command}'.format(docker=docker,
                                                            options=' '.join(docker_args),
                                                            image=image,
                                                            command=bash_string(exec_cmd))
print 'Data directory: {}'.format(data_dir)
print 'Command to run inside Docker: {}'.format(args.command)

print pull_cmd
print run_cmd
if not args.debug:
    local(pull_cmd)
    local(run_cmd)
