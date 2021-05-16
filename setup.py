import pathlib

from setuptools import find_packages, setup

import versioneer


REPO_ROOT = pathlib.Path(__file__).parent

with open(REPO_ROOT / "README.md", encoding="utf-8") as f:
    README = f.read()

REQUIREMENTS = [
    "numpy",
    "gym",
    "pandas",
    "pyarrow",
]

setup(
    name="truman",
    version=versioneer.get_version(),
    description="Simulating customer interactions with products and bid auctions at large scale.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="datavaluepeople",
    url="https://github.com/datavaluepeople/truman",
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    python_requires=">=3.7",
    cmdclass=versioneer.get_cmdclass(),
)
