from setuptools import find_packages, setup

Hypen_e_dot = "-e ."

def get_requirements(file_path:str):
    '''
    This function gives a list of requirements
    
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [item.replace("\n", "") for item in requirements]

        if Hypen_e_dot in requirements:
            requirements.remove(Hypen_e_dot)

    return requirements

setup(
    name='Student Performance ML End to End project',
    version='1.0.0',
    author = 'Vipul Tyagi',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)