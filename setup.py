from setuptools import setup

setup(
    install_requires=[
        'numpy',
        'pandas',
        'prettytable',
        'scipy',
        'scikit-learn',
        'ventmap',
        "tensorflow"
    ],
    dependency_links=[
        "https://github.com/hahnicity/ventmode/tarball/master",
        "https://github.com/hahnicity/ventmap/tarball/master"
    ]
)