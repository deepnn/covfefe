#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import sys
import os
from glob import glob
from subprocess import call

def generate():
	modules = [x.split('.')[0] for x in glob('*.py')]
	if '__init__' in modules:
		modules.remove('__init__')
	if 'generate_doc' in modules:
		modules.remove('generate_doc')
	doc_path  = '../../../docs' + \
		os.path.dirname(os.path.abspath(__file__)).split('coffee')[-1]

	for m in modules:
		cmds = 'pydoc-markdown ' + m + ' > ' + doc_path + '/' + m + '.md'
		print('Executing: {}'.format(cmds))
		os.system(cmds)

if __name__ == '__main__':
	generate()
