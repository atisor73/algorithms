from setuptools import setup, find_packages

long_description = "Algorithms is a local package for functions I keep using but am too tired to rewrite or can't seem to remember."

setup(
    name="algorithms",
    version="0.0.2",
    description="Algorithms is a local package for functions I use regularly.",
    long_description=long_description,
    license="MIT",
    author='Rosita Fu',
    author_email='rosita.fu99@gmail.com',
    packages=find_packages(),
    install_requires=['numpy','scipy','pandas'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)
