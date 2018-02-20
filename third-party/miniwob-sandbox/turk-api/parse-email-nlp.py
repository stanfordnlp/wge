#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil, re, argparse, json, csv, random, unicodedata
from codecs import open
from itertools import izip
from collections import defaultdict, Counter, OrderedDict


def clean_text(x):
    # screw non-ascii characters
    if not isinstance(x, unicode):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(ur"[‘’´`]", "'", x)
    x = re.sub(ur"[“”]", "\"", x)
    x = re.sub(ur"[‐‑‒–—−]", "-", x)
    x = re.sub(r'\s+', ' ', x).strip()
    x = str(x)
    return x


def parse_csv_record(record, storage):
    task = record['Input.task']
    for key, value in record.iteritems():
        m = re.match(r'^Answer\.(a\d+)$', key)
        if not m or not value or value == '{}':
            continue
        value = clean_text(value)
        storage[task].append(value)


# In this email inbox app, you want to find the email by <span class=red>Jill</span> and reply to her with the text <span class=red>"See you soon"</span>.
# In this email inbox app, you want to find the email by <span class=red>Jill</span> and mark it as important.
# In this email inbox app, you want to find the email by <span class=red>Jill</span> and delete it.
# In this email inbox app, you want to find the email by <span class=red>John</span> and mark it as important.
# In this email inbox app, you want to find the email by <span class=red>John</span> and forward it to <span class=red>Kate</span>.
# In this email inbox app, you want to find the email by <span class=red>Jill</span> and forward it to <span class=red>Alice</span>.
# In this email inbox app, you want to find the email by <span class=red>John</span> and reply to him with the text <span class=red>"Sounds good"</span>.
# In this email inbox app, you want to find the email by <span class=red>John</span> and delete it.

def parse_task(task):
    match = re.match(r'In this email inbox app, you want to find '
            r'the email by <span class=red>(.*)</span> and (.*)$', task)
    f_from = match.group(1)
    f_details = match.group(2)
    if f_details == 'mark it as important.':
        return ('important', (f_from, 'NAME'))
    elif f_details == 'delete it.':
        return ('delete', (f_from, 'NAME'))
    elif f_details.startswith('reply to'):
        f_message = re.search(r'<span class=red>"(.*)"</span>', f_details).group(1)
        return ('reply', (f_from, 'NAME'), (f_message, 'MSG'))
    elif f_details.startswith('forward it'):
        f_dest = re.search(r'<span class=red>(.*)</span>', f_details).group(1)
        return ('forward', (f_from, 'NAME'), (f_dest, 'DEST'))
    else:
        raise ValueError('Unrecognized: {}'.format(f_details))


def clean_msg(abstracted):
    abstracted = re.sub(r'"MSG[^ ]*"', "MSG", abstracted)
    if '"MSG"' not in abstracted:
        abstracted = abstracted.replace('MSG', '"MSG"')
    return abstracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split-size', type=float, default=.85)
    parser.add_argument('infiles', nargs='+')
    args = parser.parse_args()

    storage = defaultdict(list)
    for filename in args.infiles:
        with open(filename) as fin:
            reader = csv.DictReader(fin)
            for record in reader:
                parse_csv_record(record, storage)
    print >> sys.stderr, 'Read {} records'.format(
            sum(len(values) for values in storage.values()))

    all_templates = defaultdict(set)
    for key, values in storage.iteritems():
        task = parse_task(key)
        for value in values:
            abstracted = value
            try:
                # Replacement
                for f, t in task[1:]:
                    if f in abstracted:
                        abstracted = abstracted.replace(f, t)
                    else:
                        assert f.lower() in abstracted
                        abstracted = abstracted.replace(f.lower(), t)
                # Clean up the MSG
                if 'MSG' in abstracted:
                    abstracted = clean_msg(abstracted)
                all_templates[task[0]].add(abstracted)
            except:
                continue

    # Create a JS file
    output = {}
    for task, templates in all_templates.iteritems():
        templates = list(templates)
        random.shuffle(templates)
        num_trains = int(len(templates) * args.split_size)
        output[task] = OrderedDict([
            ('train', sorted(templates[:num_trains])),
            ('test', sorted(templates[num_trains:])),
            ])
        print >> sys.stderr, '{}: {} trains, {} test'.format(
                task, num_trains, len(templates) - num_trains)
    print 'var TEMPLATES =', json.dumps(output, indent=2, separators=(',', ':')) + ';'
    

if __name__ == '__main__':
    main()

