from setuptools import setup, find_packages


setup(
    name="realscape",
    version="0.1.0",
    description="Diffusion Models для генерации реалистичных ландшафтов",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/YOUR_USERNAME/realscape",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "diffusers>=0.24.0",
        "accelerate>=0.24.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)


