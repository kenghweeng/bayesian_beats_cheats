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
    "", ("Network Graph", "View Table", "Upload Files"))
if (options == "Network Graph"):
    st.title('Bayesian Beats Cheats - ' + options)
    st.markdown(r'<span class="red-text">Red</span> nodes are the cheaters',
                unsafe_allow_html=True)

    chosen_node_file = st.sidebar.selectbox("Choose node file to replot",
                                            ("Original File",) + tuple(os.listdir('uploads/node')))
    chosen_edge_file = st.sidebar.selectbox("Choose edge file to replot",
                                            ("Original File",) + tuple(os.listdir('uploads/edge')))
    if (st.sidebar.button("Replot") or not os.path.isfile(network_graph_html_path)):
        if (chosen_node_file == 'Original File'):
            input_node_path = input_node_file_path
        else:
            input_node_path = os.path.join("uploads", "node", chosen_node_file)
        if (chosen_edge_file == 'Original File'):
            input_edge_path = input_edge_file_path
        else:
            input_edge_path = os.path.join("uploads", "edge", chosen_edge_file)
        [num_nodes, num_edges, num_cheaters] = plot(input_node_path, input_edge_path)
        st.markdown("Number of nodes: " + str(num_nodes) +
                    " Number of edges: " + str(num_edges) + " Number of cheaters: " + str(num_cheaters))

    with open(network_graph_html_path, "r") as f:
        html = f.read()
    components.html(html, height=800)

elif (options == "View Table"):
    st.title('Bayesian Beats Cheats - ' + options)
    chosen_node_file = st.sidebar.selectbox("Choose node file to view",
                                            ("Original File",) + tuple(os.listdir('uploads/node')))
    if (chosen_node_file == 'Original File'):
        input_node_path = input_node_file_path
    else:
        input_node_path = os.path.join("uploads", "node", chosen_node_file)
    df_node = pd.read_csv(input_node_path)
    df_node['label'] = df_node.apply(lambda row: 0 if pd.isna(
        row["confessed_assignments"]) else 1, axis=1)
    st.header("Cheaters")
    df_node_cheaters = df_node[df_node['label'] == 1]
    st.dataframe(df_node_cheaters)
    st.header("Non-Cheaters")
    df_node_cheaters = df_node[df_node['label'] == 0]
    st.dataframe(df_node_cheaters)

elif (options == "Upload Files"):
    st.title("Upload Node files to plot the nodes of the network graph")
    st.write("Please ensure that the file has minimally the following headers: name, faculty, admit_year, programme, confessed_assignments")
    uploaded_node_file = st.file_uploader("Choose a file", key="node")
    if uploaded_node_file is not None:
        with open(os.path.join("uploads", "node", uploaded_node_file.name), 'wb') as f:
            f.write(uploaded_node_file.getbuffer())

    st.title("Upload edge files to plot the edges of the network graph")
    st.write("Please ensure that the file has minimally the following headers: NodeID1, NodeID2, edge_weights")
    uploaded_edge_file = st.file_uploader("Choose a file", key="edge")
    if uploaded_edge_file is not None:
        with open(os.path.join("uploads", "edge", uploaded_edge_file.name), 'wb') as f:
            f.write(uploaded_edge_file.getbuffer())
