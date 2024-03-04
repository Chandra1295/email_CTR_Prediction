from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e.'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns a list of requirements.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if not req.startswith('-e')]

    return requirements



setup(
    name='ml_email_CTR_end_to_end',
    version='0.0.1',
    author='Chandra',
    author_email='chandra.chandhan95@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )