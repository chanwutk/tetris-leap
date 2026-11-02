from setuptools import setup, find_packages

setup(
    name="leap",
    version="0.1.0",
    description="LEAP: Learning to Predict 3D Objects with an RGB-D Camera",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # Core dependencies are in requirements.txt
        # This setup.py is mainly for making the package importable
    ],
)
