from setuptools import find_packages, setup

setup(
    name="active_adaptation",
    author="btx0424@SUSTech,Haoyang Weng",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "moviepy",
        "imageio",
        "einops",
        "av", # for moviepy
        "pandas",
        "termcolor",
        "setproctitle",
        "pygame", # for game controller
        "mujoco",
        "xxhash",
        "onnxscript==0.3.0",
        "onnxruntime==1.22.0",
        "torch==2.7.0",
        "rerun_sdk<0.27",
        "torchrl==0.7.0",
        "tensordict==0.7.0",
        "numpy==1.26.0"
    ],
)
