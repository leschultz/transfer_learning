import setuptools

# Package information
name = 'transfernet'
version = '0.1.7'  # Need to increment every time to push to PyPI
description = 'Deep learning transfer learning.'
url = 'https://github.com/leschultz/transfernet.git'
author = 'Lane E. Schultz'
author_email = 'laneenriqueschultz@gmail.com'
python_requires = '>=3.10'
classifiers = ['Programming Language :: Python :: 3',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               ]
packages = setuptools.find_packages(where='src')
install_requires = [
                    'matplotlib',
                    'scikit-learn',
                    'pandas',
                    'pymatgen',
                    'numpy',
                    'pytest',
                    'torch',
                    ]

long_description = open('README.md').read()

# Passing variables to setup
setuptools.setup(
                 name=name,
                 version=version,
                 description=description,
                 url=url,
                 author=author,
                 author_email=author_email,
                 packages=packages,
                 package_dir={'': 'src'},
                 package_data={'transfernet': ['data/*']},
                 python_requires=python_requires,
                 classifiers=classifiers,
                 install_requires=install_requires,
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 )
