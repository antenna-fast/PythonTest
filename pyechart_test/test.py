import pyecharts

# TODO: 给graph加多个标签

from pyecharts import options as opts
from pyecharts.charts import Graph

# 构造数据: nodes表示节点信息和对应的节点大小; links表示节点之间的关系
nodes = [
    {"name": "结点1", "symbolSize": 10},
    {"name": "结点2", "symbolSize": 20},
    {"name": "结点3", "symbolSize": 30},
    {"name": "结点4", "symbolSize": 40},
    {"name": "结点5", "symbolSize": 50},
    {"name": "结点6", "symbolSize": 40},
    {"name": "结点7", "symbolSize": 30},
    {"name": "结点8", "symbolSize": 20},
]

links = []
# fake节点之间的两两双向关系
for i in nodes:
    for j in nodes:
        links.append({"source": i.get("name"), "target": j.get("name")})

c = (
    Graph()
    # repulsion: 节点之间的斥力因子, 值越大表示节点之间的斥力越大
    .add("", nodes, links, repulsion=8000)
    .set_global_opts(title_opts=opts.TitleOpts(title="Graph-基本示例"))
    .render("graph_base.html")
)

