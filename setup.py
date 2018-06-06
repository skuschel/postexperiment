#!/usr/bin/env python3


from setuptools import setup, find_packages
#from Cython.Build import cythonize
import numpy
import os

import versioneer

setup(name='postexperiment',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      include_package_data = True,
      author='Alexander Blinne, Stephan Kuschel',
      author_email='alexander.blinne@uni-jena.de, kuschel@stanford.edu',
      description='Postprocessor for experimental (event based) data.',
      url='https://github.com/skuschel/postexperiment',
      packages=find_packages(include=['postexperiment*']),
      include_dirs = [numpy.get_include()],
      #license='GPLv3+',
      setup_requires=['numpy>=1.8'],
      install_requires=['matplotlib>=1.3',
                        'numpy>=1.8', 'numpy>=1.9;python_version<"3.0"',
                        'scipy', 'future', 'urllib3', 'numexpr', 'functools32;python_version<"3.0"'],
      extras_require = {
        'h5 reader':  ['h5py']},
      keywords = [''],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX']
      )
