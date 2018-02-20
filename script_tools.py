import os
import tempfile
from contextlib import contextmanager
from os.path import join

import time
from fabric.api import local

from gtd.io import shell
from gtd.utils import Config


def bash_string(s):
    """Wrap a string in double quotes and escape things inside."""
    s = s.replace('\\', '\\\\')  # \ -> \\
    s = s.replace('\"', '\\\"')  # " -> \"
    return '\"{}\"'.format(s)  # s -> "s"


def check_cwd_is_repo_root():
    """Check that script is being called from root of Git repo."""
    cwd = os.getcwd()
    cwd_name = os.path.split(cwd)[1]
    if not os.path.exists(join(cwd, '.git')) or cwd_name != 'wge':
        raise RuntimeError('Must run script from root of the Git repository.')


miniwob_easy = [
    "choose-list",
    "click-button",
    "click-button-sequence",
    "click-collapsible-nodelay",
    "click-collapsible-2-nodelay",
    "click-dialog",
    "click-dialog-2",
    "click-link",
    "click-option",
    "click-pie-nodelay",
    "click-tab",
    "click-test",
    "click-test-2",
    "click-widget",
    "enter-date",
    "enter-password",
    "enter-text",
    "enter-text-dynamic",
    "focus-text",
    "focus-text-2",
    "grid-coordinate",
    "login-user",
    "navigate-tree",
    "use-autocomplete-nodelay",
    "click-color",
    "click-shape",
    "count-shape",
    "identify-shape"
]

miniwob_hard = [
    'book-flight-nodelay',
    'choose-date-nodelay',
    'search-engine',
    'guess-number',
    'click-shades',
    'tic-tac-toe',
    'enter-time',
    'click-checkboxes',
    'social-media',
    'email-inbox',
    'click-tab-2',
    'use-spinner',
]

miniwob_stochastic = [
    'click-checkboxes-large',
    'click-checkboxes-soft',
    'click-checkboxes-transfer',
    'click-tab-2-hard',
    'social-media-all',
    'social-media-some',
    'login-user-popup',
    'multi-layouts',
    'multi-orderings',
]

miniwob_nl = [
    'email-inbox-nl-turk',
]

formwob = [
    'flight.Alaska',
    'flight.Alaska-auto',
    #'flight.Alaska-auto-medium',
]

few_shot = [
    'flight.Alaska-auto',
    'enter-time',
    'social-media',
    'email-inbox-nl-turk',
    'click-tab-2-hard',
    'click-checkboxes-soft',
    'click-checkboxes-large',
]

all_unique = lambda s: len(s) == len(set(s))
assert all_unique(miniwob_easy)
assert all_unique(miniwob_hard)
assert len(set(miniwob_easy) & set(miniwob_hard)) == 0  # no overlap
assert len(miniwob_hard) + len(miniwob_easy) == 40  # total tasks

task_lists = {
    'miniwob-easy': miniwob_easy,
    'miniwob-hard': miniwob_hard,
    'miniwob-stoch': miniwob_stochastic,
    'miniwob-nl': miniwob_nl,
    'formwob': formwob,
    'few-shot': few_shot,
}


# marker class
class BundleSpec(object):
    pass


class UUIDBundleSpec(BundleSpec):
    def __init__(self, uuid):
        self.uuid = uuid


class NameBundleSpec(BundleSpec):
    def __init__(self, worksheet, name):
        self.worksheet = worksheet
        self.name = name


@contextmanager
def open_cl_file(bundle_spec, bundle_path):
    """Get the raw file content within a particular bundle at a particular path.

    Args:
        bundle_spec (BundleSpec)
        bundle_path (str): path inside the bundle. Has no leading slash.
    """

    def save_file(path):
        if isinstance(bundle_spec, UUIDBundleSpec):
            cmd = 'cl down -o {} {}/{}'.format(path,
                                               bundle_spec.uuid,
                                               bundle_path)
        elif isinstance(bundle_spec, NameBundleSpec):
            cmd = 'cl down -o {} -w {} {}/{}'.format(path,
                                                     bundle_spec.worksheet,
                                                     bundle_spec.name,
                                                     bundle_path)
        shell(cmd)

    with open_in_temp_file(save_file) as f:
        yield f


@contextmanager
def open_in_temp_file(save_file):
    """Open a file at a temporary path.
    
    Args:
        save_file (Callable[str]): takes a file path, and saves a file
            at that path.
    """
    # create temporary file just so we can get an unused file path
    f = tempfile.NamedTemporaryFile()
    f.close()  # close and delete right away
    fname = f.name

    # try to save file at the temporary path
    try:
        save_file(fname)
    except RuntimeError:
        try:
            os.remove(fname)  # if file was created, remove it
        except OSError:
            pass
        raise IOError('Failed to open file')

    # open the temporary file
    f = open(fname, 'r')
    yield f
    f.close()
    os.remove(fname)  # delete temp file


def get_metadata(bundle_spec):
    """
    
    Args:
        bundle_spec (BundleSpec)

    Returns:
        Config
    """
    path = 'data/experiments/0_unnamed/metadata.txt'
    with open_cl_file(bundle_spec, path) as f:
        f_path = os.path.realpath(f.name)
        metadata = Config.from_file(f_path)
    return metadata
