#!/usr/bin/env python
import os, sys
import numpy
from os.path import join as pjoin
import shutil
import glob

try:
    from setuptools import setup, Extension, Command
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.command.build import build
except ImportError:
    from distutils.core import setup, Extension, Command
    from distutils.command.build_ext import build_ext as _build_ext
    from distutils.command.build import build

class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = []

        for root, dirs, files in list(os.walk('pyspark')):
            for f in files:
                if f in self._clean_exclude:
                    continue
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(pjoin(root, d))

        for d in ('build', 'dist', ):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                import shutil
                shutil.rmtree(clean_tree)
            except Exception:
                pass


       

if __name__ == "__main__":

    include_dirs = ["include",
                    numpy.get_include(),
                   ]

    scripts = ['scripts/'+file for file in os.listdir('scripts/')]  

    cmdclass = {'clean': CleanCommand}
    
    try:
      import pypolychord
    except:
      print("Python bindings for PolyChordLite must be installed before mc-alf can be installed.")  
      exit()
      
    with open('routines/_version.py') as f:
        exec(f.read())

    setup(
        name = "mc-alf",
        url="https://github.com/matteofox/mc-alf",
        version= __version__,
        author="Matteo Fossati",
        author_email="matteo.fossati@durham.ac.uk",
        cmdclass = cmdclass,
        scripts = scripts, 
        packages=["routines"],
        license="LICENSE",
        description="Monte-Carlo Absorption Line Fitter",
	install_requires=[
          'numpy',
	  'scipy',
	  'matplotlib',
	  'mpi4py',
	  'linetools'],
        package_data={"": ["README.md", "LICENSE"]},
        include_package_data=True,
        zip_safe=False,
    )

