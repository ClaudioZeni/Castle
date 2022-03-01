import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    dependencies = fh.readlines()

setuptools.setup(
    name="castle",
    packages=setuptools.find_packages(),
    version="0.1",
    author="Claudio Zeni",
    author_email="czeni@sissa.it",
    description="Ensemble of ridge regression force fields in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=dependencies,
    license="Apache 2.0",
)
