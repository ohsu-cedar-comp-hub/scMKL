from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name = 'scmkl',
    version = '0.1.2',
    description = "Multiple kernel learning for single-cell data",
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Sam Kupp, Ian VanGordon, Cigdem Ak',
    author_email = 'kupp@ohsu.edu, vangordi@ohsu.edu, ak@ohsu.edu',
    packages = find_packages(),
    python_requires = '>3.11.1',
    install_requires = [
        'wheel==0.41.2',
        'anndata==0.10.8',
        'celer==0.7.3',
        'numpy==1.26.0',
        'pandas==2.2.2',
        'scikit-learn==1.3.2',
        'scipy==1.14.1'
    ]
)