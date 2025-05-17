from setuptools import setup, find_packages

setup(
    name="microflare",
    version="0.0.2",
    description="A minimalistic deep learning framework resembling PyTorch API.",
    author="Le Xuan An",
    author_email="lexuanan18102004@gmail.com",
    url="https://github.com/iSE-UET-VNU/MicroFlare",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
