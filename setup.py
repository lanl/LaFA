from setuptools import setup, find_packages


__version__ = "1.0"



# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='LaFA',
    version=__version__,
    author='Minh Vu, Ben Nebgen, Erik Skau, Geigh Zollicoffer, Juan Castorena, Kim Rasmussen, Boian Alexandrov, Manish Bhattarai',
    author_email='ceodspspectrum@lanl.gov',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/lanl/LaFA',  # change this to GitHub once published
    description=' Latent Feature Attacks on Non-negative Matrix Factorization ',
    setup_requires=['numpy', 'scipy', 'matplotlib', 'torch', 'torchvision', 'tensorly','scikit-learn', 'pytest'],
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    python_requires='>=3.11',
    classifiers=[
        'Development Status :: ' + str(__version__) + ' - Beta',
        'Programming Language :: Python :: 3.11',
        'Topic :: Machine Learning :: Libraries'
    ],
    license='License :: BSD3 License',
    zip_safe=False,
    
)