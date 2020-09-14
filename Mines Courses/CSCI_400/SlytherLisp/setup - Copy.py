# This is a script that tells pip how to setup your
# code on the system. You don't need to make any
# changes to it, nor do you need to call it yourself.
# pip runs it for you when you type commands.

from setuptools import setup

setup(
    name='slytherlisp',
    version='1.2.0-devel',
    description='Scheme like programming language for CSCI-400 at Mines',
    long_description='',
    url='https://lambda.mines.edu',
    author='Jack Rosenthal',
    author_email='jrosenth@Mines.EDU',
    license='Closed source; do not share with other students '
            'in (or who will be in) this course.',

    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='scheme',
    packages=['slyther'],
    python_requires='>=3.6, <4',
    install_requires=[
        'flake8>=3.5',
        'flake8-pep3101>=1.2.1',
        'pep8-naming>=0.7',
        'pytest>=3.8',
        'hypothesis>=3.70'],

    entry_points={
        'console_scripts': [
            'slyther=slyther.__main__:main',
        ],
    },
)
