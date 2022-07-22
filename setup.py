from setuptools import setup, find_packages


setup(
    name="bigearthnet",
    version="0.0.1",
    packages=find_packages(include=["bigearthnet", "bigearthnet.*"]),
    python_requires=">=3.8",
    install_requires=[
        "aim",
        "gdown",
        "gitpython",
        "hub",
        "hydra-core>=1.2",
        "hydra-joblib-launcher",
        "jupyter",
        "matplotlib",
        "pyyaml>=5.3",
        "pytorch_lightning==1.6.4",
        "sklearn",
        "timm",
        "torch==1.11",
        "torch_tb_profiler",
        "tqdm",
    ],
    extras_require={
        "dev": ["opencv-python"],
    },
)
