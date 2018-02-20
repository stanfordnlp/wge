#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse the results of the MiniWoB task on MTurk.

Format of the submitted demos:
Answer.d1, Answer.d2, ... are the encoded demos.
Demos are encoded as follows:
- converted to JSON
- zlib encode (to reduce size)
- base64 encode (to ensure correct transmission)
"""

import sys, os, shutil, re, argparse, json
from codecs import open
from itertools import izip
from collections import defaultdict, Counter

import csv, base64, zlib, glob, gzip


def parse_csv_record(record, args):
    workerId = record['WorkerId']
    assignmentId = record['AssignmentId']
    task = record['Input.task']
    count = 0
    used = set()
    for key, value in record.iteritems():
        m = re.match(r'^Answer\.(d\d+)$', key)
        if not m or not value:
            continue
        demo_id = m.group(1)
        if value in used:
            print u'ERROR ({}|{}|{}): Repeated demo'.format(
                    task, workerId, assignmentId)
        try:
            compressed = base64.b64decode(value)
            demo = zlib.decompress(compressed)
            assert demo[0] == '{' and demo[-1] == '}'
            count += 1
            base_dir = os.path.join(args.outdir, task)
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)
            filename = '{}_{}_{}.json.gz'.format(task, assignmentId, demo_id)
            with gzip.open(os.path.join(base_dir, filename), 'w') as fout:
                fout.write(demo)
        except Exception as e:
            print u'ERROR ({}|{}|{}): {}'.format(
                    task, workerId, assignmentId, e)
    if count != args.demos_per_hit:
        print u'WARNING ({}|{}|{}): Got {} != {} demos'.format(
                task, workerId, assignmentId,
                count, args.demos_per_hit)


def parse_json_record(record, args):
    workerId = record['metadata']['WorkerId']
    assignmentId = record['metadata']['AssignmentId']
    status = record['metadata']['AssignmentStatus']
    task = record['answers']['task']
    count = 0
    used = set()
    wtf = False
    for key, value in record['answers'].iteritems():
        m = re.match(r'^(d\d+)$', key)
        if not m or not value:
            continue
        demo_id = m.group(1)
        if value in used:
            print u'ERROR ({}|{}|{}): Repeated demo'.format(
                    task, workerId, assignmentId)
            wtf = True
        try:
            compressed = base64.b64decode(value)
            demo = zlib.decompress(compressed)
            assert demo[0] == '{' and demo[-1] == '}'
            count += 1
            base_dir = os.path.join(args.outdir, task)
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)
            filename = '{}_{}_{}.json.gz'.format(task, assignmentId, demo_id)
            with gzip.open(os.path.join(base_dir, filename), 'w') as fout:
                fout.write(demo)
        except Exception as e:
            print u'ERROR ({}|{}|{}): {}'.format(
                    task, workerId, assignmentId, e)
            wtf = True
    if count < args.demos_per_hit:
        print u'WARNING ({}|{}|{}): Got {} != {} demos'.format(
                task, workerId, assignmentId,
                count, args.demos_per_hit)
        wtf = True
    if status == 'Submitted':
        if wtf:
            print '@ BAD {} {}'.format(workerId, assignmentId)
        else:
            print '@ GOOD {} {}'.format(workerId, assignmentId)
    else:
        print '# {} {} {}'.format(status, workerId, assignmentId)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demos-per-hit', type=int, default=5,
            help='Expected number of demos per HIT')
    parser.add_argument('-o', '--outdir', default='parsed',
            help='Output directory')
    parser.add_argument('infile', nargs='+',
            help='MTurk Batch_*.csv or *.results file')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    for filename in args.infile:
        with open(filename) as fin:
            if filename.endswith('.csv'):
                reader = csv.DictReader(fin)
                for record in reader:
                    parse_csv_record(record, args)
            elif filename.endswith('.results'):
                for record in json.load(fin):
                    parse_json_record(record, args)
    

if __name__ == '__main__':
    main()
