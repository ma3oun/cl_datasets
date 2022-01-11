from setuptools import setup, find_packages

setup(
    name="cl_datasets",
    version="0.1.0",
    packages=find_packages(include=["cl_datasets", "cl_datasets.*"]),
    python_requires=">=3.8",
    install_requires=["numpy", "pillow", "torch>=1.10", "torchvision"],
    tests_require=["matplotlib>=2.2.0",],
    extras_require={"interactive": ["jupyter"],},
)
