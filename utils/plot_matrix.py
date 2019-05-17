import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    labels = ["jazz", "blues", "reggae", "pop", "disco", "country", "metal", "hip-hop", "rock", "classical"]
    matrix = np.loadtxt("../model/matrix.txt", dtype='int16')

    colors = plt.cm.get_cmap("coolwarm")

    row_colors = colors(np.linspace(0.25, 0.5, len(labels)))

    cell_colors = [[colors(i) for _ in np.linspace(0.25, 0.5, len(labels))]
                   for i in np.linspace(0.25, 0.5, len(labels))]

    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=matrix,
             cellColours=cell_colors,
             rowLabels=labels,
             rowColours=row_colors,
             colLabels=labels,
             loc='center')
    fig.tight_layout()
    plt.savefig("conf_matrix.png")
    plt.show()
