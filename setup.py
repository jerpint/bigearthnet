from setuptools import setup, find_packages


setup(
    name='bigearthnet',
    version='0.0.1',
    packages=find_packages(include=['bigearthnet', 'bigearthnet.*']),
    python_requires='>=3.8',
    install_requires=[
        'flake8',
        'flake8-docstrings',
        'gitpython',
        'hydra-core',
        'tqdm',
        'jupyter',
        'pyyaml>=5.3',
        'pytest>=4.6',
        'pytest-cov',
        'recommonmark',
        'timm',
        'torch==1.11',
        'pytorch_lightning==1.6.4'],
)
