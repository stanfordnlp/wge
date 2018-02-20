import os
from gtd.io import Workspace

# Set location of local data directory from environment variable
env_var = 'RL_DATA'
if env_var not in os.environ:
    assert False, env_var + ' environmental variable must be set.'
root = os.environ[env_var]

# define workspace
workspace = Workspace(root)
workspace.add_dir("experiments", "experiments")
workspace.add_dir("glove", "glove")
workspace.add_dir("demonstrations", "demonstrations")
