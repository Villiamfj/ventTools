from setuptools import setup

setup(
    install_requires=[
        'numpy',
        'pandas',
        'prettytable',
        'scipy',
        'scikit-learn',
        'ventmap',
        "ventmode"
    ],
    dependency_links=[
        "git+https://github.com/hahnicity/ventmode.git"
    ]
)