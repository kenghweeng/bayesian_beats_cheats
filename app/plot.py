import pandas as pd
from pyvis.network import Network


def get_hover(df):
    return df.apply(lambda row: '<strong>Name</strong>: ' + str(row['name']) + '<br><strong>Faculty</strong>: ' + str(row['faculty']) + '<br><strong>Admit Year</strong>: ' + str(row['admit_year']) + '<br><strong>Programme</strong>: ' + str(row['programme']), axis=1)


def get_color(df):
    return df.apply(lambda row: 'red' if row['label'] == 1 else '#d9f7f7', axis=1)


def plot(input_node_file_path, input_edge_file_path, out_file_path="bbc.html", height="700px", width="100%"):
    df_node = pd.read_csv(input_node_file_path)
    df_edge = pd.read_csv(input_edge_file_path)

    df_node['label'] = df_node.apply(lambda row: 0 if pd.isna(
        row["confessed_assignments"]) else 1, axis=1)

    net = Network(height="700px", width="100%")
    net.add_nodes(df_node['name'], label=df_node['name'],
                  title=get_hover(df_node), color=get_color(df_node))
    for _, row in df_edge.iterrows():
        try:
            net.add_edge(row['NodeID1'], row['NodeID2'], value=row['edge_weights'],
                         title=row['edge_weights'], physics=False)
        except AssertionError as err:
            print(err)
            pass
    net.show(out_file_path)
