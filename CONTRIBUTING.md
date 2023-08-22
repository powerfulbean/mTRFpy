# Contributions

We generally welcome all contributions.
However, please ask first before embarking on any significant pull request, otherwise you risk spending a lot of time working on something that the project's developers might not want to merge into the project.

Pull requests that seek to implement new features, should create a new branch and will be incorporated into the `main` branch once they are considered bug free and properly documented. Smaller pull requests fixing a bug may be directly merged into the `main` branch.

To make the contribution process easy and efficient we recommend that you run all style checks and the test suite, and build the documentation **locally** on your machine to detect and fix possible errors created by your changes before you submitting something. To do so, you can follow these steps:

## Setting up a development environment

For a development environment we recommend that you perform the installation in a dedicated Python environment, for example using `conda` (see: https://docs.conda.io/en/latest/miniconda.html).
Afterwards, a few additional steps need to be performed.

**For all of the steps below we assume that you work in your dedicated `mTRFpy` Python environment.**

### Install the development version of mTRFpy

Now [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the `mTRFpy` repository.
Then, `git clone` your fork and install it in "editable" mode.

```Shell
git clone https://github.com/<your-GitHub-username>/mTRFpy
cd ./mTRFpy
pip install -e ".[full]"
git config --local blame.ignoreRevsFile .git-blame-ignore-revs
```

The last command is needed for `git diff` to work properly.
You should now have the `mTRFpy` development versions available in your Python environment.

### Install additional Python packages required for compling the documentation

Navigate to the root of the `mTRFpy` repository and call:

```Shell
pip install -r doc/requirements.txt
```

This will install several packages for building the documentation for `mTRFpy`.

## Making style checks

We use [Black](https://github.com/psf/black) to format our code.
You can simply call `black .` from the root of the `mTRFpy` repository to automatically convert your code to follow the appropriate style.

Afterwards you should use [flake8](https://flake8.pycqa.org/en/latest/) to run  style checks on `mTRFpy`.
If you have accurately followed the steps to setup your `mTRFpy` development version, you can simply use the following command from the root of the `mTRFpy` repository:

```Shell
flake8 . --ignore=E501,E203,W503
```

## Running tests

We run tests using [pytest](https://docs.pytest.org/en/7.4.x/).

If you have accurately followed the steps to setup your `mTRFpy` development version, you can then simply run `pytest .` from the root of the `mTRFpy` repository

## Building the documentation

The documentation can be built using [Sphinx](https://www.sphinx-doc.org).
If you have accurately followed the steps to setup your `mTRFpy` development version,
you can simply use the following command from the root of the `mTRFpy` repository:

```Shell
sphinx-build -b html docs docs/_build
```

## Instructions for first-time contributors

When you are making your first contribution to `mTRFpy`, we kindly request you to add yourself to the CITATION.cff file.

Note: please add yourself in the "authors" section of that file, towards the end of the list of authors.
   
## Making a release

Usually only core developers make a release after consensus has been reached.
