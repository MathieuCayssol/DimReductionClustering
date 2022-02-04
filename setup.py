import setuptools

setuptools.setup(
    name='DimReduction-Clustering',
    version='0.0.0',
    author='Mathieu Cayssol',
    author_email='mathieu.cayssol@gmail.com',
    description='Dimension reduction (UMAP) + Density Based clustering (DBSCAN)',
    long_description_content_type="text/markdown",
    url='https://github.com/MathieuCayssol/DimReduction-Clustering',
    project_urls = {
        "Bug Tracker": "https://github.com/MathieuCayssol/DimReduction-Clustering/issues"
    },
    license='MIT',
    packages=['toolbox'],
    install_requires=['requests'],
)