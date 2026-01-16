from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="aqi_prediction_system",
    version="1.0.0",
    description="A comprehensive Air Quality Index prediction system",
    author="AI & Data Science Bootcamp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "aqi-collect=src.data.data_collector:main",
            "aqi-train=src.models.train_pipeline:main",
            "aqi-app=src.web.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)


