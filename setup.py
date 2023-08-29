from setuptools import setup

setup(
    name="Multiview",
    packages=[
        "generate",
        "utils",
        "ldm",
    ],
    install_requires=[
        "diffusers>=0.19.3",
        "transformers>=4.31.0",
        "einops>=0.6.1",
    ],
    author="jsp",
)
