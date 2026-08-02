from pathlib import Path

from setuptools import find_packages, setup


package_name = "team_sorting"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("tests",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/config.yaml"]),
        (
            "share/" + package_name + "/config/contracts",
            [
                "config/contracts/data_tf_policy_v1.json",
                "config/contracts/interface_v1.json",
                "config/contracts/recorder_schema_v1.json",
            ],
        ),
        *(
            [("share/" + package_name + "/docs", ["docs/data_tf_policy_v1.md"])]
            if (Path(__file__).resolve().parent / "docs/data_tf_policy_v1.md").is_file()
            else []
        ),
        ("share/" + package_name + "/launch", ["launch/team.launch.xml"]),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Team Sorting",
    maintainer_email="team@example.com",
    description="本科生机器人材料分拣竞赛的最小 ROS2 团队客户端骨架",
    url="https://example.com/team_sorting",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "team_client_node = team_sorting.ros_nodes:main_team_client",
            "perception_node = team_sorting.ros_nodes:main_perception",
            "dataset_recorder_node = team_sorting.ros_nodes:main_recorder",
        ],
    },
)
