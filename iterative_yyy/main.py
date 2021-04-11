import os
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collective.constants import get_summary_zv
from collective.Iterative import IterativeClassification

data_dir = "data/cora"
edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"),
                       sep='\t', header=None, names=["target", "source"])

Gnx = nx.from_pandas_edgelist(edgelist, create_using=nx.Graph)

feature_names = ["w_{}".format(ii) for ii in range(1433)]
column_names = feature_names + ["subject"]
node_data = pd.read_csv(os.path.join(
    data_dir, "cora.content"), sep='\t', header=None, names=column_names)
node_data['label'] = node_data['subject'].apply(
    lambda x: 1 if x == 'Theory' else 0)
node_data = node_data.loc[:, node_data.columns != 'subject']
nx.set_node_attributes(Gnx, node_data.to_dict('index'))

print("Anyhow setting edge weights")
for edge in Gnx.edges:
    Gnx[edge[0]][edge[1]]['edge_weight'] = 1

print("Getting zv")
Gnx = get_summary_zv(Gnx)

print("Getting dataframe from node attributes")
df = pd.DataFrame()
for node in Gnx.nodes:
    Gnx.nodes[node]['index'] = node
    temp = pd.DataFrame([Gnx.nodes[node]]).set_index('index')
    df = pd.concat([df, temp])

print("Doing train-test-split")
train, test = train_test_split(df, test_size=0.2)

# model1
print("Training model1")
train_x_model1 = train.drop(
    ['L1_max', 'L0_max', 'L1_mean', 'L0_mean', 'label'], axis=1)
train_y_model1 = train['label']
test_x_model1 = test.drop(
    ['L1_max', 'L0_max', 'L1_mean', 'L0_mean', 'label'], axis=1)
test_y_model1 = test['label']
model1 = LogisticRegression()
model1.fit(train_x_model1, train_y_model1)
y_pred1 = model1.predict(test_x_model1)

print(accuracy_score(test_y_model1.to_numpy(), y_pred1))

# model2
print("Training model2")
train_x_model2 = train.drop(['label'], axis=1)
train_y_model2 = train['label']
test_x_model2 = test.drop(['label'], axis=1)
test_y_model2 = test['label']
model2 = LogisticRegression()
model2.fit(train_x_model2, train_y_model2)
y_pred2 = model2.predict(test_x_model2)

print(accuracy_score(test_y_model2.to_numpy(), y_pred2))

print("Iterative classification")
ic = IterativeClassification(max_iterations=1)
new_gnx = ic.predict(Gnx, model1, model2)

new_gnx_pred = pd.DataFrame([])
for node in test['label'].index:
    temp = pd.DataFrame([[new_gnx.nodes[node]['label'][0], node]], columns=[
                        'label', 'index']).set_index('index')
    new_gnx_pred = pd.concat([new_gnx_pred, temp])

print(accuracy_score(test_y_model2.to_numpy(), new_gnx_pred.to_numpy()))
