from setuptools import setup

setup(
    name='Biomonitor',
    version='1.0',
    packages=['Biomonitor'],
    url='',
    license='GNU GPLv3',
    author='Jakub Telčer',
    author_email='',
    description='Near-real-time detections of freshwater contamination by analysis of movement of Daphnia magna',
    install_requires=['imblearn', 'matplotlib', 'numpy', 'opencv-python', 'pandas', 'scikit-learn', 'sklearn',
                      'tensorboard', 'tensorflow']
)
