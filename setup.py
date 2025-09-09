from setuptools import setup, find_packages
import re

with open("README.md") as f:
    readme = f.read()

# extract version
with open("mtrf/__init__.py") as file:
    for line in file.readlines():
        m = re.match("__version__ *= *['\"](.*)['\"]", line)
        if m:
            version = m.group(1)

setup(
    name="mtrf",
    version=version,
    description="Tools for modeling brain responses using (multivariate)"
    "temporal response functions.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="http://github.com/powerfulbean/mTRFpy",
    author="powerfulbean",
    license="MIT",
    python_requires=">=3.9",
    install_requires=["numpy", "array-api-compat"],
    extras_require={
        "testing": [
            "requests",
            "flake8",
            "black",
            "dask",
            "array-api-strict",
            "pytest",
            "tqdm",
            "matplotlib",
            "scipy",
            "black",
        ],
        "docs": ["sphinx", "sphinx_rtd_theme", "mne", "matplotlib"],
        "full": ["mtrf[testing]", "mtrf[docs]"],
    },
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
