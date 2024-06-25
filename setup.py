from setuptools import setup, find_packages

setup(
    name             = 'Drug',
    version          = '1.0.0',
    description      = 'package for distribution',
    author           = 'KANG7',
    author_email     = 'ktt0570@gmail.com',
    url              = '',
    download_url     = '',
    install_requires = ['pandas', 'numpy', 'pillow', 'rdkit', 'scikit-learn'],
	include_package_data=True,
	packages=find_packages(),
    keywords         = ['DRUGAPI', 'drugapi', 'DRUGPROCESS', 'drugprocess'],
    python_requires  = '>=3',
    zip_safe=False,
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
) 