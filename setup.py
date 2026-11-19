from setuptools import find_packages, setup

# `pip install -e .` makes `import recsys` work from any folder (notebooks, tests, the API),
# and -e (editable) means edits to src/ take effect without reinstalling.
setup(
    name="recsys",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.11",
)
