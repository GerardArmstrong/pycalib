from setuptools import setup, find_packages

# with open('README.md', 'r',encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name='pycalib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'hscimproc @ git+https://github.com/GerardArmstrong/hscimproc.git@main'
    ],
    optional_requires=['opencv-python'],
author='Gerard Armstrong')

