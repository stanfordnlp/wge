import argparse
import json
import os
from collections import OrderedDict
from os.path import join, splitext

from gtd.io import IntegerDirectories
from variational import data


parser = argparse.ArgumentParser()
parser.add_argument('run1', type=int)
parser.add_argument('run2', type=int)
args = parser.parse_args()


class Traces(OrderedDict):
    def __init__(self, d):
        items = sorted(d.items())

        for step_num, traces in items:
            assert isinstance(step_num, int)
            assert isinstance(traces, list)
            assert isinstance(traces[0], dict)

        super(Traces, self).__init__(items)


# TODO(kelvin): add 'replay' as a trace type
TRACE_TYPES = ['explore_program', 'explore_neural', 'test']


def load_trace_groups(run_num):
    """Load traces for a particular TrainingRun.
    
    Returns:
        trace_groups (dict[str, Traces]): map from trace type to Traces
    """
    run_dirs = IntegerDirectories(data.workspace.experiments)
    traces_dir = join(run_dirs[run_num], 'traces')

    trace_groups = {}
    for trace_type in TRACE_TYPES:
        trace_dir = join(traces_dir, trace_type)
        filenames = os.listdir(trace_dir)

        train_step_to_trace = {}
        for full_name in filenames:
            name, ext = splitext(full_name)
            if ext != '.json':
                continue

            full_path = join(trace_dir, full_name)
            train_step = int(name)

            with open(full_path, 'r') as f:
                trace = json.load(f)
            train_step_to_trace[train_step] = trace

        trace_groups[trace_type] = Traces(train_step_to_trace)

    return trace_groups


def fmt(collection):
    return ', '.join(str(o) for o in sorted(collection))


def trace_diff(trace1, trace2):
    trace1_extra = set(trace1) - set(trace2)
    trace2_extra = set(trace2) - set(trace1)
    overlap = sorted(set(trace1) & set(trace2))

    print 'trace1+: {}'.format(fmt(trace1_extra))
    print 'trace2+: {}'.format(fmt(trace2_extra))
    print 'overlapping keys:'
    for key in overlap:
        same = trace1[key] == trace2[key]
        same_str = 'same' if same else 'DIFFERENT'
        print '\t{}: {}'.format(key, same_str)


def traces_diff(traces1, traces2):
    # find overlapping train_steps
    overlap = sorted(set(traces1) & set(traces2))

    print 'Traces overlap on train steps: {}'.format(fmt(overlap))

    for train_step in overlap:
        print '-- STEP {} --'.format(train_step)
        print 'NOTE: only comparing first episode of each trace.'
        trace_diff(traces1[train_step][0], traces2[train_step][0])
        print


trace_groups_1 = load_trace_groups(args.run1)
trace_groups_2 = load_trace_groups(args.run2)


for trace_type in TRACE_TYPES:
    print '===== {} ====='.format(trace_type)
    traces_diff(trace_groups_1[trace_type], trace_groups_2[trace_type])
