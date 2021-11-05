import networkx as nx
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread('/Users/aibee/Desktop/1.jpeg')
G = nx.Graph()
G.add_node('KCT BS', attr_dict={'image': 'img'})
G.add_node('WAKA')
pos = nx.spring_layout(G, scale=10)
nx.draw(
    G,
    with_labels=True,
    pos=pos,
    node_size=500,
    node_color='r'
)

ax = plt.gca()
fig = plt.gcf()
trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform
imsize = 100

(x, y) = pos['KCT BS']
xx, yy = trans((x, y))
xa, ya = trans2((xx, yy))
a = plt.axes([xa - imsize / 2.0, ya - imsize / 2.0, imsize, imsize])
a.imshow(img)
a.set_aspect('equal')
a.axis('off')
plt.show()
