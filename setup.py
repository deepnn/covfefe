from setuptools import setup
from setuptools import find_packages


setup(name='coffee',
      version='0.0.1.alpha',
      description='Python wrapper for major deep learning frameworks',
      author='coffee contributers',
      author_email='esube.tamirat@gmail.com',
      url='https://github.com/deepnn/coffee',
      download_url='https://github.com/deepnn/coffee/tarball/0.0.0.alpha',
      license='MIT',
      install_requires=[
          #'theano', 
          'pyyaml', 
          'six',
          #'caffe',
          #'lasagne',
          #'tflearn',
          #'torch'
      ],
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot-ng'],
      },
      classifiers=[
          'Programming Language :: Python',
          'Operating System :: Platform Independent',
          'Intended Audience :: Developers'
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      keywords=[
          'Coffee',
          'Caffe',
          'Theano',
          'TensorFlow',
          'Deep Learning',
          'Machine Learning',
          'Neural Networks',
          'AI'
      ],
      packages=find_packages())
