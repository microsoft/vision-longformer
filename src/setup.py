#!/usr/bin/python

from setuptools import setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='msvit',
    version='0.1',
    packages=[
        'models', 'models.layers',
        # 'models.lib', 'tvm', 'tvm._ffi', 'tvm._ffi._ctypes', 'tvm.contrib'  # uncomment for cuda kernel
        ],
    # package_data={'tvm': ['*.so'], 'models': ['lib/*.so']},  # uncomment for cuda kernel
    entry_points='',
    install_requires=required,
)

