# Iterative Classification
## Algorithm
The dataset is converted into 2 vectors for each node.
``` 
1. feature_vector: numerical features of the student
   
2. feature_link_vector: numerical features and relational features which are the data from neighbors
```
### Step 1: Training
```
Train 2 K-Nearest Neighbors classifiers on training set. One for feature_vector and the other one for feature_link_vector.
```
### Step 2: Bootstrap
```
Use trained feature_vector classifier to bootstrap on test set.
```
### Step 3: Iterate (Continue till convergence)
```
1. Update relational features
    Update the [l1_avg_edge_weights, l0_avg_edge_weights, l1_max_edge_weights, l0_max_edge_weights] for all nodes
2. Classify
    Reclassify all nodes
```

## Results (K = 9)
Predicted results are in [result.csv](https://github.com/kenghweeng/bayesian_beats_cheats/blob/main/Iterative_Class/result.csv).

Classification Report

|              | precision | recall| f1-score | support |
|:------------:|:---------:|:-----:|:--------:|:-------:|
| 0            | 0.86      | 0.19  |  0.31    | 264     |
| 1            | 0.14      | 0.82  |  0.59    | 44      |
|              |           |       |          |         |
| accuracy     |           |       | 0.85     | 308     | 
| macro avg    | 0.50      | 0.50  | 0.27     | 308     |
| weighted avg | 0.76      | 0.28  | 0.30     | 308     |



