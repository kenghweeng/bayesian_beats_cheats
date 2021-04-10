# Setup

1. Install python 3.8
2. Install the python packages `pip3 install -r requirements.txt`

To see math in this readme, use Visual Studio and install the extension Markdown Math (koehlma.markdown-math)

# Toy: Cora dataset

Get the dataset and unzip to `data/cora`

```
wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
tar -xvf cora.tgz -C data
```

# Iterative Classification

Classifies a node based on its features as well as labels of neighbours

$v$: Node  
$Y_v$: Labels of node $v$  
$f_v$: feature vector of node $v$  
$z_v$: summary of labels of $v$'s neighbours (a vector)  
$\phi_1(f_v)$: predict node label based on node feature vector $f_v$  
$\phi_2(f_v, z_v)$: predict label based on node feature vector $f_v$ of labels of $v$'s neighbours

## Phase 1: Train a Classifier based on node attributes only

The classifier can be linear classifier, neural network classifier etc. This is trained on the training set to predict the labels for each node.

$\phi_1(f_v)$ : to predict $Y_v$ based on $f_v$  
$\phi_2(f_v, z_v)$ to predict $Y_v$ based on $f_v$ and summary $z_v$ of labels of $v$'s neighbours  
For vector $z_v$ of neighbourhood labels, let

- $I$ = incoming neighbour label info vector  
  $I_0$ = 1 if at least one of the incoming node is labelled 0.  
  $I_1$ = 1 if at least one of the incoming node is labelled 1.
- $O$ = outgoing neighbour label info vector  
  $O_0$ = 1 if at least one of the outgoing node is labelled 1.  
  $O_1$ = 1 if at least one of the outgoing node is labelled 1.

## Phase 2: Iterate till Convergence

On the test set, set the labels based on the classifier in Phase 1,

## Step 1: Train Classifier

On a different training set, train two classifiers:

- node attribute vector only: $\phi_1$
- node attribute and link vectors: $\phi_2$

## Step 2: Apply Classifier to test set

On test set, use trained node feature vector classifier $\phi_1$ to set $Y_v$

## Step 3.1: Update relational vectors z

Update $z_v$ for all nodes on test set

## 3.2: Update Label

Reclassify all nodes with $\phi_2$

## Iterate

Continue until convergence

- update $z_v$
- update $Y_v = \phi_2(f_v, z_v)$

# References

[http://web.stanford.edu/class/cs224w/slides/05-message.pdf](http://web.stanford.edu/class/cs224w/slides/05-message.pdf)
[https://github.com/zeno129/collective-classification/blob/master/components/collective.py](https://github.com/zeno129/collective-classification/blob/master/components/collective.py)
