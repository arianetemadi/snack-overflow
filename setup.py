from setuptools import find_packages, setup

setup(
    name="snack-overflow",
    version="0.0.1",
    description="Project for Natural Language Processing and Information Extraction course, 2024WS",
    license="MIT",
    install_requires=[
        "ipykernel==6.29.5",
        "pandas==2.2.3",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "conllu==6.0.0",
        "nltk==3.9.1",
        "wordfreq==3.1.1", 
        "scikit-learn==1.6.0",
        "torch==2.5.1",
        "numpy==1.25.0",
        "spacy==3.5.2",
        "gensim==4.3.3", 
        "transformers_interpret==0.10.0",
        "transformers==4.48.1",
        "tqdm==4.66.4",
        "absl_py==2.1.0",
        "wandb==0.18.7",
        "networkx==3.4.2",
        "pygraphviz==1.14.0"  
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
