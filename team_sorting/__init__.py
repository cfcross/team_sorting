"""物料分拣团队客户端包。

业务接口集中在 :mod:`team_sorting.interfaces`。ROS2 依赖只在
``team_sorting.ros_nodes`` 的节点启动阶段加载，因此没有比赛环境时仍可导入并测试
纯 Python 模块。
"""

__version__ = "0.1.0"
