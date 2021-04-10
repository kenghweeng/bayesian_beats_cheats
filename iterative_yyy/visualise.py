from pyvis import network as net


def draw_graph(networkx_graph, out_filename="test.html"):
    pyvis_graph = net.Network()
    for node, node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(str(node), **node_attrs)

    for source, target, edge_attrs in networkx_graph.edges(data=True):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            edge_attrs['value'] = edge_attrs['weight']
        # add the edge
        pyvis_graph.add_edge(str(source), str(target), **edge_attrs)
    pyvis_graph.show(out_filename)
