# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages, Extension
from distutils.version import LooseVersion
from os.path import join
import numpy as np

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
    raise RuntimeError("Builds without cython may be supported in future.")

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

setup(
    name='nnmnkwii',
    version='0.0.3-dev',
    description='Library to build speech synthesis systems designed for easy and fast prototyping.',
    author='Ryuichi Yamamoto',
    author_email='zryuichi@gmail.com',
    url='https://github.com/r9y9/nnmnkwii',
    license='MIT',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'numpy >= 1.8.0',
        'scipy',
        'cython >= ' + min_cython_ver,
        'bandmat',
        'fastdtw',
        'sklearn',
        'pysptk >= 0.1.7'
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
