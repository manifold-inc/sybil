from setuptools import setup, find_packages

# Read the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sybil',
    version='0.1',
    author='Manifold Technologies Inc.',
    license="BSL-1.0",
    packages=find_packages(),
    install_requires=requirements,
    # Add other arguments as necessary...
)
