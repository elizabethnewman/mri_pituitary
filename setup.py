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
    install_requires=['numpy', 'torch>=2.1.0', 'torchvision==0.16', 'Pillow', 'opencv-python', 'albumentations',
                      'matplotlib', 'pandas'],
    extras_require={'interactive': ['matplotlib']}
)
