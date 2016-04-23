try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
  name = 'nimblenet',
  packages = ['nimblenet'], # this must be the same as the name above
  version = '0.7',
  description = 'Efficient python (NumPy) neural network library.',
  long_description = 'This is an efficient implementation of a fully connected neural network in NumPy. The network can be trained by a variety of learning algorithms: backpropagation, resilient backpropagation and scaled conjugate gradient learning. The network has been developed with PYPY in mind.',
  author = 'Jorgen Grimnes',
  author_email = 'jorgenkg@yahoo.no',
  url = 'http://jorgenkg.github.io/python-neural-network/', # use the URL to the github repo
  download_url = 'https://github.com/jorgenkg/python-neural-network/tarball/0.7', # I'll explain this in a second
  keywords='python numpy neuralnetwork neural network efficient',
  install_requires = [ 'numpy' ],
  classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
)