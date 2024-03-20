from distutils.core import setup
from setuptools import find_packages

setup(
    name="autoseg",
    version="1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "gunpowder",
        "numpy",
        "scipy",
        "torch",
        "zarr",
        "daisy",
        "wandb",
        "torchvision",
        "jsonnet",
        "einops",
        "funlib.persistence",
        "funlib.geometry",
        "funlib.math",
    ],
)
