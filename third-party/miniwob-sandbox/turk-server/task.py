#!/usr/bin/env python
import argparse
import logging
import os
import random
import shutil
import sys
import time

from bottle import route, run, debug, request, static_file


################################
# Static files

@route('/flight/<path:path>')
def static_flight(path):
    """Return any static file in the files directory."""
    return static_file(path, root='static/flight')

@route('/core/<path:path>')
def static_core(path):
    """Return any static file in the files directory."""
    return static_file(path, root='static/core')


################################
# Recording demo

def get_filename():
    return (time.strftime('%m%d%H%M%S', time.gmtime()) +
            '{:.6f}'.format(random.random()) +
            '{:.6f}'.format(random.random()))

@route('/record', method='POST')
def record_demo():
    """Record demonstrations."""
    submission = request.POST
    filename = get_filename()
    with open(os.path.join('demos', filename), 'w') as fout:
        shutil.copyfileobj(request.body, fout)
    print 'Saved to {}'.format(filename)
    return filename


################################
# Launch the server

def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=8080, type=int)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-s', '--ssl', action='store_true', default=False)
    args = parser.parse_args()
    debug(args.debug)
    if args.ssl:
        import server
        run(host='0.0.0.0', port=args.port, server=server.SSLCherryPyServer)
    else:
        run(host='0.0.0.0', port=args.port)

if __name__ == '__main__':
    launch()
