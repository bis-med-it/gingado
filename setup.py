import setuptools

# Function to read requirements from a file and return a list
def read_requirements(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Parse requirements
requirements = read_requirements('requirements.txt')
dev_requirements = read_requirements('dev_requirements.txt')

setuptools.setup(
    name='gingado',
    version='0.2.0',
    description='A machine learning library for economics and finance',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    author='Douglas K. G. de Araujo',
    author_email='Douglas.Araujo@bis.org',
    license='Apache Software License 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    url='https://github.com/bis-med-it/gingado',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements
    },
    python_requires='>=3.7',
    zip_safe=False,
    entry_points={},
)

