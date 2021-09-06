import setuptools

setuptools.setup(
    name='HyperspectralTrios',
    version='0.0.1',
    # packages=setuptools.find_packages()
    packages=['HyperspectralTrios'],
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.17',
        'pandas>=0.24',
        'pyodbc>=4.0',
        'openpyxl==3.0.7',
        'plotly>=4.10',
        'jupyterlab'
    ]
)

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

# setuptools.setup(
#     name="example-pkg-YOUR-USERNAME-HERE",
#     version="0.0.1",
#     author="Example Author",
#     author_email="author@example.com",
#     description="A small example package",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/pypa/sampleproject",
#     project_urls={
#         "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
#     },
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
#     package_dir={"": "."},
#     packages=setuptools.find_packages(where="src"),
#     python_requires=">=3.6",
# )

