# -*- coding: utf-8 -*-
from setuptools import find_namespace_packages, setup, find_packages

from pipestonks import (
    __author__,
    __author_email__,
    __description__,
    __license__,
    __package_name__,
    __long_description__,
    __url__,
    __version__,
)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt") as f:
    test_requirements = f.read().splitlines()


setup(
    name=__package_name__,
    author=__author__,
    author_email=__author_email__,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    description=__description__,
    install_requires=requirements,
    include_package_data=True,
    license=__license__,
    long_description_content_type="text/markdown",
    long_description=__long_description__,
    # packages=find_namespace_packages(
    #     "Weekstab", exclude=["docs", "tests", "workflows"]
    # ),
    packages=find_packages(exclude=("configs", "tests*", "scripts")),
    tests_require=test_requirements,
    url=__url__,
    version=__version__,
    python_requires=">=3.8",
)
