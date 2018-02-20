#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil, re, argparse, json, time, traceback, urlparse
from codecs import open
from itertools import izip
from collections import defaultdict, Counter

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_options(headless=False):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('headless')
        options.add_argument('disable-gpu')
        options.add_argument('no-sandbox')
    return options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--screenshot',
            help='dump screenshot to this directory')
    parser.add_argument('-H', '--headless', action='store_true',
            help='do not render the Chrome interface')
    parser.add_argument('-n', type=int, default=30,
            help='number of episodes')
    parser.add_argument('-b', '--base-url',
            default='http://localhost:8000/miniwob/',
            help='base URL of the task')
    args = parser.parse_args()

    if args.screenshot:
        assert os.path.isdir(args.screenshot)

    print 'Opening Chrome'
    options = get_options(headless=args.headless)
    driver = webdriver.Chrome(chrome_options=options)
    driver.implicitly_wait(1)
    url = urlparse.urljoin(args.base_url, 'click-test-2.html')
    print 'Go to {}'.format(url)
    driver.get(url)

    for i in xrange(args.n):
        print 'Instance {}'.format(i)
        element = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.ID, "sync-task-cover")))
        driver.find_element_by_id('sync-task-cover').click()
        if args.screenshot:
            driver.save_screenshot(os.path.join(args.screenshot, '{}.png'.format(i)))
        buttons = driver.find_elements_by_tag_name('button')
        one_buttons = [x for x in buttons if x.text == 'ONE']
        clicked = False
        for x in one_buttons:
            try:
                x.click()
                clicked = True
                break
            except Exception as e:
                # May fail if TWO is in front of ONE!
                traceback.print_exc()
        if not clicked:
            print 'Attempt to click other things'
            for x in buttons:
                try:
                    x.click()
                    clicked = True
                    break
                except Exception as e:
                    traceback.print_exc()
        print 'Cliked = {}'.format(clicked)
            
    print 'DONE!'
    driver.close()
    

if __name__ == '__main__':
    main()

