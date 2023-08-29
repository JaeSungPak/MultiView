from setuptools import setup

setup(
    name="Multiview",
    packages=[
        "generate",
        "utils"
    ],
    install_requires=[
        "diffusers>=0.19.3",
        "transformers>=4.31.0",
    ],
    author="jsp",
)
