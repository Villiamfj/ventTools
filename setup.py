from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'prettytable',
        'scipy',
        'scikit-learn',
        'ventmap',
        "ventmode"
    ]
)