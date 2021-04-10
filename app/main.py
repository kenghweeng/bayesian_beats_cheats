import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from IPython.core.display import display, HTML

st.set_page_config(page_title="Bayesian Beats Cheats", layout="wide")
con1 = st.beta_container()
con1.header('Bayesian Beats Cheats')
graph = nx.Graph()
graph.add_edge('first', 'second')
graph.add_edge('second', 'third')


def draw_graph(networkx_graph, out_filename="test.html"):
    pyvis_graph = Network(
        height="700px", width="100%")
    pyvis_graph.from_nx(networkx_graph)
    pyvis_graph.show(out_filename)


draw_graph(graph)

with open("test.html", "r") as f:
    html = f.read()
components.html(html, height=800)

# display(HTML("test.html"))
