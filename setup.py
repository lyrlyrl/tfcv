import sys
import os

from setuptools import setup, find_packages

version = '1.0.0'

project_name = 'tfcv'

long_description = ' '
if '--dev' in sys.argv:
    project_name = project_name+'-dev'
    sys.argv.remove('--dev')

def _get_requirements():
    """Parses requirements.txt file."""
    install_requires_tmp = []
    dependency_links_tmp = []

    with open(
        os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as f:
        for line in f:
            package_name = line.strip()
            # Skip empty line or comments starting with "#".
            if not package_name or package_name[0] == '#':
                continue
            if package_name.startswith('-e '):
                dependency_links_tmp.append(package_name[3:].strip())
            else:
                install_requires_tmp.append(package_name)

    if os.environ.get('INSIDE_DOCKER', False):
        install_requires_tmp.append('opencv-python-headless')
    else:
        install_requires_tmp.append('opencv-python')
        
    return install_requires_tmp, dependency_links_tmp

install_requires, dependency_links = _get_requirements()

setup(
    name=project_name,
    version=version,
    description='Custom Implementions',
    long_description=long_description,
    author='Yiren Lin',
    author_email='yirenlin1125@google.com',
    url='https://github.com/lyrlyrl/tfcv.git',
    license='Apache 2.0',
    packages=find_packages(exclude=[
        'configs',
    ]),
    exclude_package_data={
        '': ['*_test.py',],
    },
    install_requires=install_requires,
    dependency_links=dependency_links,
    python_requires='>=3.8',
)
