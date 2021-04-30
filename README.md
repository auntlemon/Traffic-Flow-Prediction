# Temporal Multi-Graph Convolutional Network for Traffic Flow Prediction

The project is a Tensorflow implementation of T-MGCN proposed in the paper “Temporal Multi-Graph Convolutional Network for Traffic Flow Prediction”. T-MGCN is a multi-graph convolutional network designed for predicting traffic flow in a road network.

The paper can be visited at https://ieeexplore.ieee.org/document/9098104

## Requirement

1. cuda = 9.0
2. cudnn = 7.0
3. python = 3.6
4. tensorflow-gpu = 1.8.0

## Data Preparatio

We implement and execute T-MGCN on two real traffic flow datasets, i.e., HZJTD and PEMSD10.

**(1)   HZJTD**

HZJTD was collected from Hangzhou Integrated Transportation Research Center. It was sampled from 202 roads in the major urban areas of Hangzhou, China. Before running the model, the following files should be prepared.

* data/hz/graph.pkl: This file is used to create the topological graph *Gr*.
* data/hz/similar_matrix_hz.txt: This file is used to create the traffic pattern graph *Gp*.
* data/hz/cosine_matrix_hz.txt: This file is used to create the functionality graph *Gf*.
* data/hz/road_geo.txt: This is an optional file used to execute the command “python create_dis.py”.
* data/hz/norm_matrix_hz.txt: This file contains the normalized traffic flow samples.

**(2)   PEMSD10**

PEMSD10 is a subset of an open traffic flow dataset, PEMS, collected by California Department of Transportation. Before running the model, the following files should be prepared.

* data/pems/similar_distance_pems.txt: This file is used to create the topological graph *Gr*.
* data/pems/similar_matrix_pems.txt: This file is used to create the traffic pattern graph *Gp*.
* data/pems/norm_matrix_pems.txt: This file contains the normalized traffic flow samples.
* data/incident_d.txt & gcn/data/incident_meta.txt: These are optional files used to execute the command “python train_pems_incident.py” to extract the event features.
* data/glove.6B: This is an optional file that contains the word embeddings based on GloVe and is used to support the command “python train_pems_incident.py”.

## Run the Project

### Run the command below to train the model

**(1)**  **train on the HZJTD dataset:**

* python create_dis.py: This is an optional command, which is used to create the weighted topological graph *Gw*.
* **python train_hz.py**: This is the command used to train the model.
* After running the training program, it would generate the following files.
* data/test_hz.txt: This file contains the testing samples.
* checkpoints_distance_adj_n5h: This file contains the trained model.

**(2)**  **train on the PEMSD10 dataset:**

* **python train_pems.py**: This is the command used to train the model.
* python train_pems_incident.py: This is the command used to train the model by considering the event features.

After running the training program, it would generate the following files.

* data/test_pems.txt: This file contains the testing samples.
* checkpoints_pems_adj_n1h: This file contains the trained model.
* checkpoints_pems_cnn_text_n4h_1: This file contains the trained model by considering the event features.

### Run the command below to test the model

**(1)**  **test on the HZJTD dataset:**

python test_hz.py: This is the command used to test the model.

After running the testing program, it would generate the following files.

data/n5h_hz_dis_adj_label.txt: This file contains the ground truth.

data/n5h_hz_dis_adj_pre.txt: This file contains the prediction results.

**(2)**  **test on the PEMSD10 dataset:**

* python test_pems.py: This is the command used to test the model.
* python test_pems_incident.py: This is the command used to test the model by considering the event features.

After running the testing program, it would generate the following files.

* data/pems_n5h_label.txt: This file contains the ground truth.
* data/pems_n5h_pre.txt: This file contains the prediction results.
* data/pems_cnn_text_n5h_label.txt: This file contains the ground truth by considering the event features.
* data/pems_cnn_text_n5h_pre.txt: This file contains the prediction results by considering the event features.







