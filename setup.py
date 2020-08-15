from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='mlff',
      version='0.1',
      description='Machine learning based metallic force fields',
      url='https://github.com/hauser-group/mlff',
      author='Ralf Meyer',
      author_email='meyer.ralf@yahoo.com',
      packages=['mlff', 'mlff.models'],
      install_requires=requirements)
