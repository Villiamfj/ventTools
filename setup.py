from setuptools import setup

setup(
    install_requires=[
        'numpy',
        'pandas',
        'prettytable',
        'scipy',
        'scikit-learn',
        'ventmap'
    ],
    dependency_links=[
        "https://github.com/hahnicity/ventmode/tarball/master"
    ]
)