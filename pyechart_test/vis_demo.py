from pyecharts import options as opts
from pyecharts.charts import Graph
import os


vis_demo_path = '/ssd/linchen/buc3.0/debug/all_in_one/'

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
for i in nodes:
    for j in nodes:
        links.append({"source": i.get("name"),
                      "target": j.get("name")})

c = (
    Graph()
    .add("",
         nodes,
         links,
         repulsion=800)
    .set_global_opts(title_opts=opts.TitleOpts(title="Graph-基本示例"))
    .render(vis_demo_path + "graph_base.html")
)

