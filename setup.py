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
    python_requires=">=3.8",
    install_requires=["numpy"],
    extras_require={"testing": ["pytest", "tqdm", "matplotlib", "scipy"]},
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
