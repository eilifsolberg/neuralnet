try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description' : 'My Project',
    'author' : 'Eilif Solberg',
    'url' : 'URL to get it at.',
    'download_url' : 'Where to download it.',
    'author_email' : 'eilifsolberg@gmail.com',
    'version' : '0.1',
    'install_requires' : ['nose'],
    'packages' : ['neuralnet'],
    'scripts' : [],
    'name' : 'projectname'
}

setup(**config)
