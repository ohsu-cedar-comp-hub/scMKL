from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name = 'scmkl',
    version = '0.5.0a',
    description = "Single-cell analysis using Multiple Kernel Learning",
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Sam Kupp, Ian VanGordon, Cigdem Ak',
    author_email = 'kupp@ohsu.edu, vangordi@ohsu.edu, ak@ohsu.edu',
    url = 'https://github.com/ohsu-cedar-comp-hub/scMKL/tree/main',
    packages = find_packages(),
    python_requires = '>=3.11.1, <=3.13.13',
    install_requires = [
        'wheel',
        'anndata',
        'celer',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'numba',
        'plotnine',
        'matplotlib',
        'scanpy',
        'umap-learn',
        'muon',
        'gseapy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ]
)