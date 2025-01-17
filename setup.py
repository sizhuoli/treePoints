from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='treepoints',
    version='0.1.0',
    description='A Python library with a TensorFlow model for building a tree database from remote sensing data.',
    author='Sizhuo Li',
    author_email='sizli@ign.ku.dk',
    url='https://github.com/sizhuoli/treePoints',
    package_dir={'': 'treePoints'},
    packages=find_packages(where='treePoints'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'tensorflow[and-cuda]==2.15.1',
        'keras==2.15.0',
    ],
    python_requires='>=3.11',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: GIS',
    ],
)