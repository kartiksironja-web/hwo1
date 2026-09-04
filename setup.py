
from setuptools import find_packages, setup

package_name = 'hw01_tf_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kartik',
    maintainer_email='kartik.sironja@iitgn.ac.in',
    description='HW01 TF current frame vs fixed frame rotation demo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rotation_demo = hw01_tf_demo.rotation_demo:main',
        ],
    },
)
