from setuptools import setup

setup(name='ti_nbody',
      version='0.1',
      description='Nbody Simulation in Taichi',
      url='https://github.com/xuyanwen2012/ti_nbody',
      author='Yanwen Xu',
      author_email='yxu83@ucsc.edu',
      license='MIT',
      install_requires=['taichi>=0.7.1', 'numpy>=1.20'],
      packages=['ti_nbody', 'ti_nbody/algorithms'],
      zip_safe=False)
