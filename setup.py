from setuptools import setup, find_packages

setup(
    name="jpeg-antecedent",
    version="0.1.0",
    author="Etienne Levecque",
    author_email="pythonrepo.outbid878@silomails.com",
    description="Find possible antecedents of JPEG compression and decompression when the JPEG pipeline is known.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EtienneLevecque/jpeg-antecedent",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy==2.2.0",
        "pillow==11.0.0",
        "scikit-image==0.25.0",
        "jpeglib==1.0.1",
        "scipy==1.14.1",
        "gurobipy==11.0.3",
        "PyYAML==6.0.2",
        "rich==13.7.1",
        "h5py~=3.10.0",
        "pandas==2.2.3"
    ],
)
