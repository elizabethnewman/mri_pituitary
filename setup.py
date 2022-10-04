from setuptools import setup, find_packages

setup(
    name='mri_pituitary',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/elizabethnewman/mri_pituitary',
    license='MIT',
    author='Elizabeth Newman',
    author_email='elizabeth.newman@emory.edu',
    description='',
    install_requires=['numpy', 'torch', 'torchvision', 'Pillow', 'opencv-python', 'albumentations'],
    extras_require={'interactive': ['matplotlib']}
)
