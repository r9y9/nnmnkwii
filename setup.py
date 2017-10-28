# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages, Extension
import setuptools.command.develop
import setuptools.command.build_py
from distutils.version import LooseVersion
from os.path import join, exists
import subprocess
import os
import numpy as np

version = '0.0.9'

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv('NNMNKWII_BUILD_VERSION'):
    version = os.getenv('NNMNKWII_BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except subprocess.CalledProcessError:
        pass


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, cwd
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'nnmnkwii', 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):

    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


def create_readme_rst():
    global cwd
    try:
        subprocess.check_call(
            ["pandoc", "--from=markdown", "--to=rst", "--output=README.rst",
             "README.md"], cwd=cwd)
        print("Generated README.rst from README.md using pandoc.")
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


min_cython_ver = '0.21.0'
try:
    import Cython
    ver = Cython.__version__
    _CYTHON_INSTALLED = ver >= LooseVersion(min_cython_ver)
except ImportError:
    _CYTHON_INSTALLED = False

try:
    if not _CYTHON_INSTALLED:
        raise ImportError('No supported version of Cython installed.')
    from Cython.Distutils import build_ext
    cython = True
except ImportError:
    cython = False

if cython:
    ext = '.pyx'
    cmdclass = {'build_ext': build_ext}
else:
    ext = '.c'
    cmdclass = {}
    print("Building extentions from pre-generated C source")
    if not os.path.exists(join("nnmnkwii", "paramgen", "mlpg_helper" + ext)):
        raise RuntimeError("Cython is required to generate C code.")

ext_modules = [
    Extension(
        name="nnmnkwii.util._linalg",
        sources=[join("nnmnkwii", "util", "_linalg" + ext)],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=["-std=c99"],
    ),
    Extension(
        name="nnmnkwii.paramgen.mlpg_helper",
        sources=[join("nnmnkwii", "paramgen", "mlpg_helper" + ext)],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=["-std=c99"]
    ),
]

cmdclass['build_py'] = build_py
cmdclass['develop'] = develop

if not exists('README.rst'):
    create_readme_rst()

if exists('README.rst'):
    README = open('README.rst', 'rb').read().decode("utf-8")
else:
    README = ''

setup(
    name='nnmnkwii',
    version=version,
    description='Library to build speech synthesis systems designed for easy and fast prototyping.',
    long_description=README,
    author='Ryuichi Yamamoto',
    author_email='zryuichi@gmail.com',
    url='https://github.com/r9y9/nnmnkwii',
    license='MIT',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'numpy >= 1.11.0',
        'scipy',
        'cython >= ' + min_cython_ver,
        'bandmat',
        'fastdtw',
        'sklearn',
        'pysptk >= 0.1.7',
        'tqdm',
    ],
    tests_require=['nose', 'coverage'],
    extras_require={
        'docs': ['numpydoc', 'sphinx_rtd_theme'],
        'test': ['nose', 'pyworld', 'librosa'],
    },
    classifiers=[
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    keywords=["Research"]
)
