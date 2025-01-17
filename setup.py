from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='treepoints',
    version='0.1.6',
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
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: GIS',
    ],
    keywords='tree detection, remote sensing, deep learning',
    project_urls={
    'Source': 'https://github.com/sizhuoli/treePoints',
    'Documentation': 'https://github.com/sizhuoli/treePoints/README.md'
    }

)