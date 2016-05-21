try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
  name             = 'nimblenet',
  packages         = ['nimblenet'], # this must be the same as the name above
  version          = '0.1',
  description      = 'Efficient python (NumPy) neural network library.',
  long_description = 'This is an efficient implementation of a fully connected neural network in NumPy. The network can be trained by a variety of learning algorithms: backpropagation, resilient backpropagation and scaled conjugate gradient learning. The network has been developed with PYPY in mind.',
  author           = 'Jorgen Grimnes',
  author_email     = 'jorgenkg@yahoo.no',
  url              = 'https://jorgenkg.github.io/python-neural-network/',
  download_url     = 'https://github.com/jorgenkg/python-neural-network/tarball/0.1',
  keywords         = ["python", "numpy", "neuralnetwork", "neural", "network", "efficient", "lightweight"],
  install_requires = [ 'numpy' ],
  extras_require   = {
          'efficient_sigmoid'            : ["scipy"],
          'training_with_scipy_minimize' : ["scipy"]
      },
  classifiers      = [
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],

)