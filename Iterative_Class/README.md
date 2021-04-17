# Iterative Classification
## Algorithm
The dataset is converted into 2 vectors for each node.
``` 
1. feature_vector: 
   [name, admit_year, participation, pe, finals, total, percentile, 
   afast, pe_percent, finals_percent, midterms, midterms_percent, level,
   level_min_max, exp, exp_min_max, num_videos, avg_videos_completion,
   t01_exp, t02_exp, t03_exp, t04_exp, t05_exp, t06_exp, t07_exp,
   t08_exp, t09_exp, t10_exp, num_confessed_assignments]
   
2. feature_link_vector: 
   [name, admit_year, participation, pe, finals, total, percentile, 
   afast, pe_percent, finals_percent, midterms, midterms_percent, level,
   level_min_max, exp, exp_min_max, num_videos, avg_videos_completion,
   t01_exp, t02_exp, t03_exp, t04_exp, t05_exp, t06_exp, t07_exp,
   t08_exp, t09_exp, t10_exp, num_confessed_assignments,
   input_1_avg_edge_weights, input_0_avg_edge_weights,
   input_1_max_edge_weights, input_0_max_edge_weights]
   
 [name, admit_year, participation, pe, finals, total, percentile, 
 afast, pe_percent, finals_percent, midterms, midterms_percent, level,
 level_min_max, exp, exp_min_max, num_videos, avg_videos_completion,
 t01_exp, t02_exp, t03_exp, t04_exp, t05_exp, t06_exp, t07_exp,
 t08_exp, t09_exp, t10_exp, num_confessed_assignments] are the numerical features of the student
 
 [l1_avg_edge_weights, l0_avg_edge_weights, l1_max_edge_weights, l0_max_edge_weights] are relational features which are the data from neighbors
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

## Results
Predicted results are in [result.csv](https://github.com/kenghweeng/bayesian_beats_cheats/blob/main/Iterative_Class/result.csv).

The Classification Report is as follows:
K = 9
Ratio of training/test set: 8:2
|              | precision | recall| f1-score | support |
|:------------:|:---------:|:-----:|:--------:|:-------:|
| 0            | 0.89      | 0.99  |  0.94    | 212     |
| 1            | 0.00      | 0.00  |  0.00    | 27      |
|              |           |       |          |         |
| accuracy     |           |       | 0.88     | 239     | 
| macro avg    | 0.44      | 0.50  | 0.47     | 239     |
| weighted avg | 0.79      | 0.88  | 0.83     | 239     |
