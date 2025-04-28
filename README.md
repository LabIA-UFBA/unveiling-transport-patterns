# Unveiling Patterns in Public Transportation: A Graph-Based Deep Learning Approach

## 🗺️ Overview

The research provides an integrated public transportation dataset from Salvador (Brazil), along with machine learning benchmarks designed to model and predict mobility patterns. The dataset captures daily movements of over 710,000 passengers, covering around 2,000 vehicles and nearly 400 lines across 3,000 stops and stations. In addition to offering raw and processed data, we provide experimental code and models that leverage spatiotemporal features to reproduce and extend the results described in our study.



<center><img src="images/graphs-SSA.png" width=500px/></center>





---

## 🗂️ Repository structure

The organization of this repository is:

> - **data** : raw and graph-based data
> - **data_design** : contains source codes developed to create our datasets and train learning models 
> - **images**: plots with statistics of the dataset attributes
> - **models** : contains frozen models used to predict different tasks using SUNT
> - **outputs**/: weigths and results
---


##  ▶️  Benchmarks (Reproducing Experiments)

### Requirements

To install requirements:

```
pip install -r requirements.txt
```

### Node Regression

All code in `models/node-regression/main.py`

**Load data**: Enter in dir `models/node-regression/` :

```python
# train data - node features (split with PyTorch Geometric Temporal)
train_dataset = load_dataset('../../data/graph_designer/train_test/dataset_train.pkl')
# test data - node features (split with PyTorch Geometric Temporal)
test_dataset = load_dataset('../../data/graph_designer/train_test/dataset_test.pkl')
# selected test nodes
df_nodes = pd.read_csv('../../data/graph_designer/train_test/df_nodes_selected.csv')
nodes = list(df_nodes.tensor_idx.values)
df_nodes_loader = pd.read_csv('../../data/graph_designer/train_test/df_nodes_selected_loader.csv')
df_nodes_loader['time'] = pd.to_datetime(df_nodes_loader['time'], format='%Y-%m-%d %H:%M:%S')
```

**Train and test**:

```python
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
hidden_layer = 128
mn = 'gcn'
epcs=500
# instance model
model =  mm.GCN(in_channels=36,
                hidden_channels=hidden_layer,
                out_channels=12).to(device)

# run train and test model
run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device)
```

---

### Node classification

All code in `models/node-classification/main.py`

**Load data**: Enter in dir `models/node-classification/` :

```python
# load train and test by fold
data = torch.load(f'../../data/graph_designer/train_test_node_classification_days/data_{fold_idx}.pt')
```

**Train and Test**

```python
# ser parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_layer  = data.num_features
hidden_layer = 64
out_layer    = 1#data.y.shape[1]
mn = f'gcn{sufix}'
epcs=100

# instance a model
model =  mm.GCN(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer).to(device)
# train and test
run_model(model, data, mn, epcs, device=device)
```

---

### Edge classification


All code in `models/edge-classification/main.py`

**Load data**: Enter in dir `models/edge-classification/` :

```python
# load train and test by fold
train_dataset = torch.load(f'../../data/graph_designer/train_test_edge_classification_days/train_data_{fold_idx}.pt')
        test_dataset  = torch.load(f'../../data/graph_designer/train_test_edge_classification_days/test_data_{fold_idx}.pt')
```

**Train and Test**

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_layer  = train_dataset.num_features
hidden_layer = 64
out_layer    = train_dataset.edge_label.shape[1]
mn = f'gcn{sufix}'
epcs=50
    
# instance a model
model =  mm.GCNEdgeClassifier(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer).to(device)
# traind and test
run_model(model, train_dataset, test_dataset, mn, epcs, device=device)
```

## 📃 License

This project is licensed under the MIT License.