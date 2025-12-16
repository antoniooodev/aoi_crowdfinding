from setuptools import setup, find_packages

setup(
    name="aoi_crowdfinding",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    author="Antonio",
    description="AoI-Aware Crowd-Finding Game Theory Simulation",
)
