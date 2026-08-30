from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path, encoding='utf-8') as file_obj:
        requirements = [
            requirement.strip()
            for requirement in file_obj
            if requirement.strip() and not requirement.lstrip().startswith('#')
        ]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='mlproject',
version='0.0.1',
author='Pavel',
author_email='koripave@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
)
