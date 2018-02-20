#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil, re, argparse, json, gzip
from codecs import open
from itertools import izip
from collections import defaultdict, Counter



EMAIL_INBOX_PATTERNS = [
        ('delete', r'Find the email by (.*) and click the trash icon to (.*) it\.', ['by', 'task']),
        ('forward', r'Find the email by (.*) and (.*) that email to (.*)\.', ['by', 'task', 'to']),
        ('important', r'Find the email by (.*) and click the (.*) icon to mark it as important\.', ['by', 'task']),
        ('reply', r'Find the email by (.*) and (.*) to them with the text "(.*)"\.', ['by', 'task', 'message']),
        ]

def extract_email_inbox(utterance):
    for task, regex, keys in EMAIL_INBOX_PATTERNS:
        match = re.match(regex, utterance)
        if match:
            return dict(zip(keys, match.groups()))
    raise ValueError('Bad email-inbox utterance: {}'.format(utterance))


NL_TEMPLATES = [
  'Find the email by (?P<NAME>[^ ]*) and forward that email to (?P<DEST>[^ ]*).',
  'Locate the email by (?P<NAME>[^ ]*). Forward that email to (?P<DEST>[^ ]*).',
  'Look for the email from (?P<NAME>[^ ]*) and forward to (?P<DEST>[^ ]*).',
  'Forward to (?P<DEST>[^ ]*) the email from (?P<NAME>[^ ]*).',
  'Send (?P<DEST>[^ ]*) the email you got from (?P<NAME>[^ ]*).',
  'Go to the email by (?P<NAME>[^ ]*). Send it to (?P<DEST>[^ ]*).',
  'Send to (?P<DEST>[^ ]*) the email you got from (?P<NAME>[^ ]*).',
  'Forward the email from (?P<NAME>[^ ]*) to (?P<DEST>[^ ]*).',
  'Forward to (?P<DEST>[^ ]*) the email from (?P<NAME>[^ ]*).',
  'Send (?P<DEST>[^ ]*) the email from (?P<NAME>[^ ]*).',
  'Please find the message by (?P<NAME>[^ ]*), then send it to (?P<DEST>[^ ]*).',
  'Please forward the information from (?P<NAME>[^ ]*) to (?P<DEST>[^ ]*).',
  '(?P<DEST>[^ ]*) wants the email you got from (?P<NAME>[^ ]*).',
  '(?P<DEST>[^ ]*) wants the email (?P<NAME>[^ ]*) sent to you.',
  'The mail by (?P<NAME>[^ ]*) should be forwarded to (?P<DEST>[^ ]*).',
  'Please forward to (?P<DEST>[^ ]*) the email by (?P<NAME>[^ ]*).',
  'Give (?P<DEST>[^ ]*) the message you received from (?P<NAME>[^ ]*),',
  'Forward the mail by (?P<NAME>[^ ]*) to (?P<DEST>[^ ]*).',
  'Go to the message from (?P<NAME>[^ ]*) and send it to (?P<DEST>[^ ]*).',
  '(?P<DEST>[^ ]*) is waiting for the email by (?P<NAME>[^ ]*).',
  '(?P<NAME>[^ ]*) wants his or her message to be sent to (?P<DEST>[^ ]*).',
  'I want the mail by (?P<NAME>[^ ]*) to be sent to (?P<DEST>[^ ]*).',
  'Forward to (?P<DEST>[^ ]*) the email you got from (?P<NAME>[^ ]*).',
  'Please forward the message from (?P<NAME>[^ ]*) to (?P<DEST>[^ ]*).',
  'Please find the mail by (?P<NAME>[^ ]*). Forward it to (?P<DEST>[^ ]*).',
  'Navigate to the message from (?P<NAME>[^ ]*) and send it to (?P<DEST>[^ ]*).',
  'Forward (?P<DEST>[^ ]*) the email from (?P<NAME>[^ ]*).',
  'Forward (?P<DEST>[^ ]*) the message (?P<NAME>[^ ]*) sent you.',
  'Send (?P<DEST>[^ ]*) the information (?P<NAME>[^ ]*) sent to you.',
  'Search for the mail (?P<NAME>[^ ]*) sent you and send it to (?P<DEST>[^ ]*).',
  ]
NL_TEMPLATES = [re.compile(x) for x in NL_TEMPLATES]

def extract_email_inbox_forward_nl(utterance):
    for regex in NL_TEMPLATES:
        match = regex.match(utterance)
        if match:
            return {
                'by': match.group('NAME'),
                'to': match.group('DEST'),
                }
    raise ValueError('Bad email-inbox utterance: {}'.format(utterance))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['all', 'forward'])
    parser.add_argument('filename')
    args = parser.parse_args()

    with gzip.open(args.filename) as fin:
        data = json.load(fin)
    utterance = data['utterance']
    if args.mode == 'all':
        data['fields'] = extract_email_inbox(utterance)
    elif args.mode == 'forward':
        data['fields'] = extract_email_inbox_forward_nl(utterance)


    outfile = args.filename.replace('.json.gz', '-fixed.json.gz')
    with gzip.open(outfile, 'w') as fout:
        json.dump(data, fout, separators=(',', ':'))
    print >> sys.stderr, '{} -> {}'.format(args.filename, outfile)

if __name__ == '__main__':
    main()

