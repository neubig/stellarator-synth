#!/usr/bin/env python3
"""Setup script for stellarator synthetic data generation package."""

from setuptools import setup, find_packages

with open("README_synthetic_data.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stellarator-synth",
    version="0.1.0",
    author="OpenHands AI",
    author_email="openhands@all-hands.dev",
    description="Synthetic data generation for stellarator simulator ML surrogates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neubig/stellarator-synth",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "stellarator-synth-generate=synthetic_data_generator:main",
            "stellarator-synth-train=ml_surrogate_model:main",
        ],
    },
)