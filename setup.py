from setuptools import setup

setup(
    name="Multiview",
    packages=[
        "generate",
        "utils",
        "ldm",
        "ldm.models.diffusion",
        "ldm.modules.diffusionmodules",
        "ldm.modules.distributions",
        "ldm.modules.encoders",
        "ldm.thirdp.psp",
        "ldm.models",
        "ldm.modules",
        "ldm.thirdp",
        "download",
    ],
    install_requires=[
        "diffusers>=0.19.3",
        "transformers>=4.31.0",
        "einops>=0.6.1",
        "omegaconf>=2.3.0",
        "onnxruntime>=1.15.1",
        "onnx>=1.14.0",
        "segment_anything@git+https://github.com/facebookresearch/segment-anything.git#egg=onnx,onnxruntime",
        "clip@git+https://github.com/openai/CLIP.git",
        "rembg>=2.0.50",
        "trimesh>=3.23.1",
        "pytorch-lightning>=2.0.6",
        "matplotlib>=3.7.2",
        "rich>=13.5.2",
        "kornia>=0.7.0",
       ],
    author="jsp",
)
