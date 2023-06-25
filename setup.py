from setuptools import setup

setup(
    name="abcsmc",
    version="0.0.1",
    packages=["abcsmc"],
    package_dir={'abcsmc': 'abcsmc'},
    entry_points={
        "console_scripts": [
            "abcsmc = abcsmc.main:main"
        ]
    },
)
