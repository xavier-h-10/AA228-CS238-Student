import matplotlib.pyplot as plt
import networkx as nx


def read_gph_file(file_path):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Read the file and add edges to the graph
    with open(file_path, 'r') as file:
        for line in file:
            node1, node2 = line.strip().split(',')
            G.add_edge(node1, node2)

    return G


# Assuming the file is saved in the path '/mnt/data/graph.gph'
file_path = 'small.gph'
G = read_gph_file(file_path)

# Draw the graph using a circular layout
plt.figure(figsize=(8, 8))
nx.draw_circular(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold',
                 arrowstyle='->', arrowsize=20)
plt.savefig("small.pdf")
plt.show()
