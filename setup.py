#!/usr/bin/env python
from distutils.core import setup
import setuptools

with open("requirements.txt") as f:
    requirements = list(map(lambda x: x.strip(), f.read().strip().splitlines()))

setup(
    name="sts_jax",
    version="0.1",
    description="JAX code for structural time series",
    url="https://github.com/probml/sts-jax",
    install_requires=requirements,
    packages=setuptools.find_packages(),
)
