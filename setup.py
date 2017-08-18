# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages

setup(
    name='nnmnkwii',
    version='0.0.2',
    description='Library to build speech synthesis systems designed for easy and fast prototyping.',
    author='Ryuichi Yamamoto',
    author_email='zryuichi@gmail.com',
    url='https://github.com/r9y9/nnmnkwii',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.8.0',
        'scipy',
        'bandmat',
        'fastdtw',
        'sklearn',
        'pysptk >= 0.1.7'
    ],
    tests_require=['nose', 'coverage'],
    extras_require={
        'docs': ['numpydoc', 'sphinx_rtd_theme'],
        'test': ['nose', 'pyworld'],
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
