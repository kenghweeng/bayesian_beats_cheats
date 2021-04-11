import os
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import pandas as pd
from pyvis.network import Network

st.set_page_config(page_title="Bayesian Beats Cheats", layout="wide")


def hover_html(node_attr):
    html_text = ""
    for key in node_attr.keys():
        if (key != "size"):
            html_text = html_text + "<strong>" + \
                str(key) + "</strong>: " + str(node_attr[key]) + "<br>"
    return html_text


def draw_graph(networkx_graph, out_filename="test.html"):
    pyvis_graph = Network(
        height="700px", width="100%")
    pyvis_graph.from_nx(networkx_graph)
    for node in networkx_graph.nodes:
        vis_node = pyvis_graph.get_node(node)
        vis_node['title'] = hover_html(networkx_graph.nodes[node])
        if (node != "Dawson" and node != "Eugene"):
            vis_node['color'] = "red"
    pyvis_graph.show(out_filename)


st.markdown("""
<style>
.red-text {
    color: red !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Choose your display")
options = st.sidebar.selectbox(
    "", ("Network Graph", "Flagged Cheaters"))
if (options == "Network Graph"):
    st.title('Bayesian Beats Cheats - ' + options)
    st.markdown(r'<span class="red-text">Red</span> nodes are the cheaters',
                unsafe_allow_html=True)
    if (st.sidebar.button("Replot") or not os.path.isfile("test.html")):
        graph = nx.Graph()
        graph.add_edge('Alice', 'Bob')
        graph.add_edge('Bob', 'Charlie')
        graph.add_edge('Dawson', 'Charlie')
        graph.add_edge('Dawson', 'Eugene')

        for node in graph.nodes:
            graph.nodes[node]['Assignment'] = 'Runes'

        draw_graph(graph)
    with open("test.html", "r") as f:
        html = f.read()
    components.html(html, height=800)
elif (options == "Flagged Cheaters"):
    st.title('Bayesian Beats Cheats - ' + options)

    df_cheaters = pd.DataFrame([["Alice", "Bob", "Runes"], [
        "Charlie", "Bob", "Runes"]], columns=["Name 1", "Name 2", "Assignment"])
    st.dataframe(df_cheaters)
