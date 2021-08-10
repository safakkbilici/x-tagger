from setuptools import setup, find_packages


setup(
    name = "x-tagger",
    packages=find_packages(),
    version = "0.1.5",
    license = "MIT",
    description = "A Natural Language Processing toolkit for token classification in its simplest form.",
    author = "Şafak Bilici",
    author_email = "safakk.bilici.2112@gmail.com",
    url = "https://github.com/safakkbilici/x-tagger",
    download_url = "https://github.com/safakkbilici/x-tagger/archive/refs/tags/0.1.5.tar.gz",
    keywords = [
        "Hidden Markov Models",
        "Token Classification",
        "Deep Learning For Token Classification"
    ],
    install_requires=[
        "datasets>=1.6",
        "torch>=1.6",
        "transformers>=4.0",
        "torchtext"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
