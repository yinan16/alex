# ----------------------------------------------------------------------
# Created: tis feb 23 22:32:09 2021 (+0100)
# Last-Updated:
# Filename: setup.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alex-nn", # Replace with your own username
    version="0.1.1",
    author="Yinan Yu",
    author_email="yu.yinan16@gmail.com",
    description="Network analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yinan16/alex",
    project_urls={
        "Bug Tracker": "https://github.com/yinan16/alex/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        "pandas",
        "numpy",
        "pydot==1.4.1",
        "pyparsing",
        "scikit-learn==0.23.2",
        "PyYAML>=5.4",
        "jsonschema",
        "rope",
        "matplotlib",
    ],
    scripts=["bin/alex-nn"],
    include_package_data=True,
    package_dir={"": "./"},
    packages=["alex",
              "alex.components",
              "alex.annotators",
              "alex.engine",
              "alex.alex"],
    package_data={'alex.components': ['*.yml']},
    python_requires=">=3.6",
)
