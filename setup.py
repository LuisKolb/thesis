from setuptools import setup, find_packages

setup(
    name="thesis-lkolb",
    version="0.1.0",
    description="Thesis: Fact-Checking Claims using Authority Retrieval",
    author="Luis Kolb",
    author_email="kolb.luis@gmail.com",
    url="https://luiskolb.at",
    install_requires=[
        # "pyserini",
        "jupyter",
        # "faiss-cpu",
        "scikit-learn",
        "python-terrier",
        "ir-measures",
        "jsonlines",
        "openai",
        "plotly",
        "matplotlib",
        "transformers",
        "sentence-transformers"
    ],
    packages=find_packages(".", ["lkae/*"]),
)
