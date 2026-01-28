#!/usr/bin/env python
import os, sys
import numpy
from os.path import join as pjoin
import shutil
import glob

from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build import build


class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = []

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

    if sys.version_info[0] < 3:
      raise Exception("This codes requires Python3")

    scripts = ['scripts/mc-alf']  
    
    cmdclass = {'clean': CleanCommand}
      
    with open('mcalf/_version.py') as f:
        exec(f.read())

    setup(
        name = "mc-alf",
        url="https://github.com/matteofox/mc-alf",
        version= __version__,
        author="Matteo Fossati",
        author_email="matteo.fossati@unimib.it",
        cmdclass = cmdclass,
        scripts = scripts, 
        packages=["mcalf",
                  "mcalf.routines"],
        license="LICENSE",
        description="Monte-Carlo Absorption Line Fitter",
	install_requires=[
          'numpy',
	  'scipy',
	  'matplotlib',
	  'linetools'],
        package_data={"": ["README.md", "LICENSE"]},
        include_package_data=True,
        zip_safe=False,
    )

