import os
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import pandas as pd
from pyvis.network import Network
from plot import plot

network_graph_html_path = "bbc.html"
input_node_file_path = "../data/unified_node_data.csv"
input_edge_file_path = "../data/uniq_lines_edge_weights.csv"

st.set_page_config(page_title="Bayesian Beats Cheats", layout="wide")

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
    if (st.sidebar.button("Replot") or not os.path.isfile(network_graph_html_path)):
        [num_nodes, num_edges] = plot(
            input_node_file_path, input_edge_file_path)
        st.markdown("Number of nodes: " + str(num_nodes) +
                    " Number of edges: " + str(num_edges))
    with open(network_graph_html_path, "r") as f:
        html = f.read()
    components.html(html, height=800)
elif (options == "Flagged Cheaters"):
    st.title('Bayesian Beats Cheats - ' + options)
    df_node = pd.read_csv(input_node_file_path)
    df_node['label'] = df_node.apply(lambda row: 0 if pd.isna(
        row["confessed_assignments"]) else 1, axis=1)
    st.dataframe(df_node.sort_values(by=['label'], ascending=False))
