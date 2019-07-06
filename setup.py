# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages, Extension
import setuptools.command.develop
import setuptools.command.build_py
from distutils.version import LooseVersion
from setuptools.command.build_ext import build_ext as _build_ext
from os.path import join, exists
import subprocess
import os
import sys

version = '0.0.19'

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
    except IOError:  # FileNotFoundError for python 3
        pass


class build_ext(_build_ext):
    # https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


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


cmdclass = {"build_py": build_py, "develop": develop}

min_cython_ver = '0.28.0'
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

include_dirs = []
cmdclass['build_ext'] = build_ext
if cython:
    ext = '.pyx'
    import numpy as np
    include_dirs.insert(0, np.get_include())
else:
    ext = '.c'
    print("Building extentions from pre-generated C source")
    if not os.path.exists(join("nnmnkwii", "paramgen", "mlpg_helper" + ext)):
        raise RuntimeError("Cython is required to generate C code.")

ext_modules = [
    Extension(
        name="nnmnkwii.util._linalg",
        sources=[join("nnmnkwii", "util", "_linalg" + ext)],
        include_dirs=include_dirs,
        language="c",
        extra_compile_args=["-std=c99"],
    ),
    Extension(
        name="nnmnkwii.paramgen.mlpg_helper",
        sources=[join("nnmnkwii", "paramgen", "mlpg_helper" + ext)],
        include_dirs=include_dirs,
        language="c",
        extra_compile_args=["-std=c99"]
    ),
]


def package_files(directory):
    # https://stackoverflow.com/questions/27664504/
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


package_data = package_files("./nnmnkwii/util/_example_data")

install_requires = [
    'scipy',
    'cython >= ' + min_cython_ver,
    'fastdtw',
    'sklearn',
    'pysptk >= 0.1.17',
    'tqdm',
]

# Workaround for python 3.7 and bandmat
# https://github.com/r9y9/deepvoice3_pytorch/issues/154
if sys.version_info[:2] != (3, 7):
    install_requires.append('bandmat >= 0.7')

setup(
    name='nnmnkwii',
    version=version,
    description='Library to build speech synthesis systems designed for easy and fast prototyping.',
    long_description=open('README.md', 'rb').read().decode("utf-8"),
    long_description_content_type='text/markdown',
    author='Ryuichi Yamamoto',
    author_email='zryuichi@gmail.com',
    url='https://github.com/r9y9/nnmnkwii',
    license='MIT',
    packages=find_packages(),
    package_data={'': package_data},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    setup_requires=["numpy >= 1.11.0"],
    install_requires=install_requires,
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
