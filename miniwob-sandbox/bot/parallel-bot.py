#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil, re, argparse, json
from codecs import open
from itertools import izip
from collections import defaultdict, Counter

from selenium import webdriver
from selenium.webdriver.common.keys import Keys


def main():
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()

    drivers = []
    for i in xrange(5):
        drivers.append(webdriver.Remote(
            desired_capabilities={'browserName': 'chrome'}))
    print 'Created 5 drivers'

    for driver in drivers:
        driver.get("https://github.com")
        print(driver.title)
        assert "GitHub" in driver.title

    for driver in drivers:
        driver.close()
    

if __name__ == '__main__':
    main()

