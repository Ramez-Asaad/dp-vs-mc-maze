from setuptools import setup, find_packages

with open("README_GITHUB.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="reinforcement-learning-maze",
    version="1.0.0",
    author="Student",
    description="Dynamic Programming vs Monte Carlo on Maze Navigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/reinforcement-learning-maze",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
)
