from setuptools import setup

setup(
    name="Multiview",
    packages=[
        "generate",
        "utils",
        "ldm",
        "ldm.models.diffusion",
        "ldm.modules.diffusionmodules",
    ],
    install_requires=[
        "diffusers>=0.19.3",
        "transformers>=4.31.0",
        "einops>=0.6.1",
        "omegaconf>=2.3.0",
        "onnxruntime>=1.15.1",
        "onnx>=1.14.0",
        "rembg>=2.0.50",
        
    ],
    author="jsp",
)
