from setuptools import setup, find_packages


setup(
    name = "x-tagger",
    packages = find_packages(exclude=["examples"]),
    version = "0.1.0",
    license = "MIT",
    description = "A Natural Language Processing toolkit for token classification in its simplest form.",
    author = "Şafak Bilici",
    author_email = "safakk.bilici.2112@gmail.com",
    url = "https://github.com/safakkbilici/x-tagger",
    keywords = [
        "Hidden Markov Models",
        "Token Classification",
        "Deep Learning For Token Classification"
    ],
    install_requires=[
        "datasets>=1.6",
        "torch>=1.6",
        "transformers>=4.0",
        "torchtext",
    ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Development Status :: Beta'
    ],
)
