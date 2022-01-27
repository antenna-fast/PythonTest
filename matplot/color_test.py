import numpy as np
import matplotlib.pyplot as plt


cmaps = {}
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
# cmaps['Miscellaneous'] = [
#             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
#             'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
#             'gist_ncar']


# all color
import matplotlib
for name, hex in matplotlib.colors.cnames.items():
    print('name: ', name, hex)


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list):
    # Create figure and adjust figure height to number of colormapsvvv
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99)
    axs[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name, i in zip(axs, cmap_list, [i for i in range(100)]):
        # ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.imshow(gradient, aspect='auto', cmap=plt.cm.gist_rainbow_r(i))
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps.items():
    plot_color_gradients(cmap_category, cmap_list)

plt.show()
