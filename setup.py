import setuptools

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="tracer",
    version='0.1.6',
    author='hgryu',
    author_email='hgryu@euclidsoft.co.kr',
    description="Module responsible for preprocessing data for network analysis",
    py_modules=['tracer'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: MIT",
        "Operating System :: OS Independent",
    ],
)
