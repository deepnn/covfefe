#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import sys
import os
from shutil import copyfile

def copy():
    src = './README.md'
    dst= './docs/index.md'
    copyfile(src, dst)

if __name__ == '__main__':
    copy()
