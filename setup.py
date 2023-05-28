from setuptools import setup

setup(
   name = 'Nesting Problem',
   url = 'https://github.com/jonielbarreto/nestingProblem',
   author = 'Joniel B. Barreto',
   author_email = 'joniel.bb@gmail.com',
   packages = ['nestingProblem'],
   install_requires = ['numpy'],
   version = '0.1.0',
   license = 'MIT',
   description = 'Some heuristics for the Nesting Problem',
   long_description = open('README.txt').read(),
)
