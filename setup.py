from setuptools import setup, find_packages


setup(name='boiles',
      version='0.1',
      description='a package that helps building ILES scheme using Bayesian optimization based on gpytorch, botorch and ax',
      url='http://github.com/fengyiqi/boiles.git',
      author='yiqi',
      author_email='yiqi.feng@hotmail.com',
      license='Apache',
      packages=find_packages(),
      )