from setuptools import setup

with open("README.md", "r") as f:
    desc = f.read()

setup(
    name="portan",
    version="0.1.0",
    description="Portfolio analytics and reporting toolset in Python",
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/draktr/portan",
    author="draktr",
    license="MIT License",
    packages=["portan"],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "yfinance",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "statsmodels",
    ],
    keywords="portfolio finance asset-management quant trading investment",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering",
    ],
    project_urls={
        "Documentation": "https://portan.readthedocs.io/en/latest/",
        "Issues": "https://github.com/draktr/portan/issues",
    },
)
