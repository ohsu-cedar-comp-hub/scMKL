from setuptools import setup, find_packages

setup(
    name = 'scmkl',
    version = '0.1.1',
    description = "Multiple kernel learning for single-cell data",
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