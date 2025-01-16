from setuptools import setup, find_packages

with open('scr/README.md', 'r') as f:
    long_description = f.read()


setup(
    name='treepoints',
    version='0.1.0',
    description='A Python library with a TensorFlow model for building a tree database from remote sensing data.',
    author='Sizhuo Li',
    author_email='sizli@ign.ku.dk',
    url='',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'python=3.11',
        'pip=24',
        'pillow=10.3.0',
        'rasterio=1.3.10',
        'imgaug=0.4.0',
        'scikit-learn=1.5.0',
        'ipython=8.25.0',
        'tensorflow[and-cuda]==2.15.1',
    ],
    python_requires='>=3.6',
)