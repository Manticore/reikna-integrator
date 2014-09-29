from setuptools import setup, find_packages

VERSION = '0.1.0'
RELEASE = '0.1.0+dev'


if __name__ == '__main__':

    DOCUMENTATION = open('README.rst').read()

    setup(
        name='reikna-integrator',
        packages=find_packages(),
        namespace_packages=['reiknacontrib'],
        install_requires=[
            ('reikna >= 0.6.4'),
            ('progressbar2 >= 2.6.0'),
            ],
        version=VERSION,
        author='Bogdan Opanchuk',
        author_email='bogdan@opanchuk.net',
        url='http://github.com/Manticore/reikna-integrator',
        description='A collection of SDE integration tools based on Reikna',
        long_description=DOCUMENTATION,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Topic :: Scientific/Engineering',
            'Operating System :: OS Independent'
        ]
    )
