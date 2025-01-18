from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'uwb_simulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adren',
    maintainer_email='andresmarsil77@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_simulator = uwb_simulator.trajectory_simulator:main',
            'odometry_simulator = uwb_simulator.odometry_simulator:main',
            'measurement_simulator_eliko = uwb_simulator.measurement_simulator_eliko:main',
            'clock_publisher = uwb_simulator.clock_publisher:main'
        ],
    },
)
