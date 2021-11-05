import matplotlib.pyplot as plt


fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

for i in range(10):
    ax.scatter(i, i,  alpha=0.5)
    # plt.scatter(i, i, label=str(i))
    # plt.legend()
    plt.savefig('/Users/aibee/PycharmProjects/pythonProject/test/matplot/123.png')

    ax2.scatter(i, i**2, alpha=0.6)
    plt.savefig('/Users/aibee/PycharmProjects/pythonProject/test/matplot/456.png')

ax.set_xlabel(r'$\Delta_i$', fontsize=15)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
ax.set_title('Volume and percent change')

# ax.grid(True)
# fig.tight_layout()

plt.show()
