from setuptools import setup, find_packages

setup(
    name="zynkode-compute",
    version="0.1.0",
    author="Zara Quinn",
    author_email="zara.quinn@zynkode.dev",
    description="GPU compute orchestration for AMD ROCm/HIP",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24.0", "pyyaml>=6.0", "rich>=13.0"],
)
