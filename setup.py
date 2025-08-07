from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name = 'scmkl',
    version = '0.3.0',
    description = "Single-cell analysis using Multiple Kernel Learning",
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Sam Kupp, Ian VanGordon, Cigdem Ak',
    author_email = 'kupp@ohsu.edu, vangordi@ohsu.edu, ak@ohsu.edu',
    url = 'https://github.com/ohsu-cedar-comp-hub/scMKL/tree/main',
    packages = find_packages(),
    python_requires = '>=3.11.1, <3.13',
    install_requires = [
        'wheel==0.41.2',
        'anndata==0.10.8',
        'celer==0.7.3',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.5.1',
        'scipy==1.14.1',
        'numba==0.61.2',
        'plotnine==0.14.3',
        'matplotlib==3.9.3'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ]
)