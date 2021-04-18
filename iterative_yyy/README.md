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

# References

[http://web.stanford.edu/class/cs224w/slides/05-message.pdf](http://web.stanford.edu/class/cs224w/slides/05-message.pdf)
[https://github.com/zeno129/collective-classification/blob/master/components/collective.py](https://github.com/zeno129/collective-classification/blob/master/components/collective.py)
