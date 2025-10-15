import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import KFold
from math import sqrt
from sklearn.metrics import mean_squared_error
import sklearn
import data_xy
from sklearn.preprocessing import StandardScaler
import sys
from PySide6.QtWidgets import QLineEdit, QCheckBox, QFrame, QTextEdit, QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QDialog,QGridLayout, QMessageBox
from PySide6 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtGui import QIcon, QFont
from icon import MyIcon
from Err import error
#import tensorflow as tf
import csv
from PySide6.QtCore import Qt
import os
import subprocess

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')

class CustomGraphDataset(torch.utils.data.Dataset):
        def __init__(self, X, edge_index, edge_attr, y):
            self.data_list = [Data(x=X[i], edge_index=edge_index[i], edge_attr=edge_attr[i], y=y[i]) for i in range(len(X))]
        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]

class CustomGraphDataset_p(torch.utils.data.Dataset):
        def __init__(self, X, edge_index, edge_attr):
            self.data_list = [Data(x=X[i], edge_index=edge_index[i], edge_attr=edge_attr[i]) for i in range(len(X))]
        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]
        
        

def custom_collate(data_list):
        # Use Batch from PyTorch Geometric to collate Data objects into a batch
        return Batch.from_data_list(data_list)
    
def one_hot_encoding(x, permitted_list):
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        """

        if x not in permitted_list:
            x = permitted_list[-1]

        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

        return binary_encoding
def get_atom_features(atom, 
                          use_chirality = True, 
                          hydrogens_implicit = True):
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """

        # define list of permitted atoms
        
        permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
        
        if hydrogens_implicit == False:
            permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
        
        # compute atom features
        
        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        
        n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        
        formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        
        hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        
        is_in_a_ring_enc = [int(atom.IsInRing())]
        
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                        
        if use_chirality == True:
            chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc
        
        if hydrogens_implicit == True:
            n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens_enc

        return np.array(atom_feature_vector)

def get_bond_features(bond, 
                          use_stereochemistry = True):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """

        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

        bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        
        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
        if use_stereochemistry == True:
            stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc
       # print(bond_feature_vector)
        return np.array(bond_feature_vector)
def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
        """
        Inputs:
            
        
        x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
        y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
        
        Outputs:
        
        data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
        
        """

        
        data_list = []
        for (smiles, y_val) in zip(x_smiles, y):
            try:
                # convert SMILES to RDKit mol object
                mol = Chem.MolFromSmiles(smiles)
                # get feature dimensions
                n_nodes = mol.GetNumAtoms()
                n_edges = 2*mol.GetNumBonds()
                unrelated_smiles = "O=O"
                unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
                n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
                n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
                # construct node feature matrix X of shape (n_nodes, n_node_features)
                X = np.zeros((n_nodes, n_node_features))
                for atom in mol.GetAtoms():
                    X[atom.GetIdx(), :] = get_atom_features(atom)               
                X = torch.tensor(X, dtype = torch.float)
                # construct edge index array E of shape (2, n_edges)
                (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
                torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
                torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
                E = torch.stack([torch_rows, torch_cols], dim = 0)       
                # construct edge feature array EF of shape (n_edges, n_edge_features)
                EF = np.zeros((n_edges, n_edge_features))   
                for (k, (i,j)) in enumerate(zip(rows, cols)):            
                    EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))    
                EF = torch.tensor(EF, dtype = torch.float) 
                # construct label tensor
                y_tensor = torch.tensor(np.array([y_val]).reshape(-1,1), dtype = torch.float)  
                # construct Pytorch Geometric data object and append to data list
                data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
            except (TypeError, AttributeError):
              #  print(f'Invalid SMILES: {smiles}')
                continue
        return data_list

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels_p(x_smiles):
        """
        Inputs:
            
        
        x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
        y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
        
        Outputs:
        
        data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
        
        """

        
        data_list = []
        for smiles in x_smiles:
            try:
                # convert SMILES to RDKit mol object
                mol = Chem.MolFromSmiles(smiles)
                # get feature dimensions
                n_nodes = mol.GetNumAtoms()
                n_edges = 2*mol.GetNumBonds()
                unrelated_smiles = "O=O"
                unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
                n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
                n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
                # construct node feature matrix X of shape (n_nodes, n_node_features)
                X = np.zeros((n_nodes, n_node_features))
                for atom in mol.GetAtoms():
                    X[atom.GetIdx(), :] = get_atom_features(atom)               
                X = torch.tensor(X, dtype = torch.float)
                # construct edge index array E of shape (2, n_edges)
                (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
                torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
                torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
                E = torch.stack([torch_rows, torch_cols], dim = 0)       
                # construct edge feature array EF of shape (n_edges, n_edge_features)
                EF = np.zeros((n_edges, n_edge_features))   
                for (k, (i,j)) in enumerate(zip(rows, cols)):            
                    EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))    
                EF = torch.tensor(EF, dtype = torch.float) 

                # construct Pytorch Geometric data object and append to data list
                data_list.append(Data(x = X, edge_index = E, edge_attr = EF))
            except (TypeError, AttributeError):
              #  print(f'Invalid SMILES: {smiles}')
                continue
        return data_list


class GNN(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.init_ui()

    def  init_ui(self):
        self.pixmappath = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
        self.node_all1 =[]
        self.layer_all1 = []
        self.epochs_all = []
        self.batch_all = []
        self.label_all1 = []

        self.NL = 0
      #  self.setGeometry(200, 100, 50, 50)
        self.setWindowTitle('GNN Molecule')
        MyIcon(self)
        font = QFont()
        font.setBold(True)
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        self.label2 = QLabel('Extra Features', self)
        self.label2.setFont(font)
        self.grid_layout.addWidget(self.label2, 0, 0)
        
        self.label3 = QLabel('Parametes', self)
        self.label3.setFont(font)
        self.grid_layout.addWidget(self.label3, 6, 0)
        
        self.label4 = QLabel('Layers', self)
        self.label4.setFont(font)
        self.grid_layout.addWidget(self.label4, 11, 0)
        
        
        self.Atom_type = QCheckBox("Atom type")
        self.grid_layout.addWidget(self.Atom_type, 2, 0)
        
        self.Atom_type = QCheckBox("No. heavy neighbor")
        self.grid_layout.addWidget(self.Atom_type, 2, 1)
        
        self.Atom_type = QCheckBox("Formal charge")
        self.grid_layout.addWidget(self.Atom_type, 2, 2)
        
        self.Atom_type = QCheckBox("Hybridisation type")
        self.grid_layout.addWidget(self.Atom_type, 3, 0)
        
        self.Atom_type = QCheckBox("Is in Ring")
        self.grid_layout.addWidget(self.Atom_type, 3, 1)
        
        
        self.Atom_type = QCheckBox("Is Aromatic")
        self.grid_layout.addWidget(self.Atom_type, 3, 2)
        
        self.Atom_type = QCheckBox("Atomic Mass")
        self.grid_layout.addWidget(self.Atom_type, 4, 0)
        
        self.Atom_type = QCheckBox("VdW radius")
        self.grid_layout.addWidget(self.Atom_type, 4, 1)
        
        self.Atom_type = QCheckBox("Covalant radius")
        self.grid_layout.addWidget(self.Atom_type, 4, 2)
        
        self.labelOpt = QLabel('Optimizer ' , self) 
        self.grid_layout.addWidget(self.labelOpt, 8, 0)
        self.combo_box1 = QComboBox()
        self.combo_box1.addItems(['adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'AdamW','SparseAdam','LBFGS' ])
        self.grid_layout.addWidget(self.combo_box1, 8, 1)

        self.epochs_in = QLabel('No. Epochs ' , self) 
        self.grid_layout.addWidget(self.epochs_in, 9, 0)
        self.line_edit1 = QLineEdit(self)
        self.grid_layout.addWidget(self.line_edit1, 9, 1) 
        
        
     #   self.ep = QLineEdit(self)
     #   self.grid_layout.addWidget(self.ep, 2, 1)
      #  self.epochs_all.append(self.ep)
        
        self.batchs_in = QLabel('Batch size ' , self) 
        self.grid_layout.addWidget(self.batchs_in, 10, 0)
        
 #       self.label_1 = QLabel('From :', self)
 #       self.grid_layout.addWidget(self.label_1, 10, 1)
        self.line_edit_1 = QLineEdit(self)
        self.grid_layout.addWidget(self.line_edit_1, 10, 1)
  #      self.label_2 = QLabel('to :', self)
  #      self.grid_layout.addWidget(self.label_2, 10, 3)
  #      self.line_edit_2 = QLineEdit(self)
  #      self.grid_layout.addWidget(self.line_edit_2, 10, 4)
  #      self.label_3 = QLabel('step :', self)
  #      self.grid_layout.addWidget(self.label_3, 10, 5)
  #      self.line_edit_3 = QLineEdit(self)
  #      self.grid_layout.addWidget(self.line_edit_3, 10, 6)
        
   #     self.ba = QLineEdit(self)
   #     self.grid_layout.addWidget(self.ba, 3, 1)
   #     self.batch_all.append(self.ba)
               # self.combo_box1_all.append(self.combo_box1.currentText())

              #  layout = QGridLayout()
               # self.setLayout(layout)

                

        
           
        self.button1 = QPushButton('New Layer', self)
        self.button1.clicked.connect(self.run_svr)
        self.grid_layout.addWidget(self.button1, 13, 0)
        
        
        separator = QFrame(self)
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        
        separator1 = QFrame(self)
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        
        separator0 = QFrame(self)
        separator0.setFrameShape(QFrame.HLine)
        separator0.setFrameShadow(QFrame.Sunken)
        
        separator01 = QFrame(self)
        separator01.setFrameShape(QFrame.HLine)
        separator01.setFrameShadow(QFrame.Sunken)
        
        # add the separator to the grid layout
        self.grid_layout.addWidget(separator01, 1, 0, 1,8)
        self.grid_layout.addWidget(separator0, 7, 0, 1,8)
        self.grid_layout.addWidget(separator1, 12, 0, 1,8)
     #   self.grid_layout.addWidget(separator, 14, 0, 1,8)
        
        
                
        self.label00 = QLabel('Number of Nodes', self)
        self.label00.setFont(font)
        self.grid_layout.addWidget(self.label00, 14, 1)
        self.label01 = QLabel('Select the Function', self)
        self.label01.setFont(font)
        self.grid_layout.addWidget(self.label01, 14, 3)        
        
        # set the alignment of the labels
        self.label00.setAlignment(Qt.AlignCenter)
        self.label01.setAlignment(Qt.AlignCenter)
        
        # add the labels to the grid layout
        self.grid_layout.addWidget(self.label00, 14, 1)
        self.grid_layout.addWidget(self.label01, 14, 3)
        
        self.button1 = QPushButton('Run', self)
        self.button1.clicked.connect(self.run_svr2)
        self.grid_layout.addWidget(self.button1, 0, 6)     
        self.n = 14
        
        self.delete_button = QPushButton('Delete', self)
        self.grid_layout.addWidget(self.delete_button, self.n , 5)
        self.delete_button.clicked.connect(self.delete_layer)





     #   self.n += 1
        self.mm = self.n
    def run_svr(self):
        self.NL += 1
        self.label1 = QLabel('Layer No. ' + str(self.NL), self)
        self.grid_layout.addWidget(self.label1, self.n + 1, 0)
        self.label_all1.append(self.label1)
        
        self.s = QLineEdit(self)
        self.grid_layout.addWidget(self.s, self.n + 1, 1)
        self.node_all1.append(self.s)
        
        self.combo_box = QComboBox()
        self.combo_box.addItems(['relu', 'sigmoid', 'tanh', 'deserialize', 
                                 'elu', 'exponential', 'gelu', 'get', 
                                 'hard_sigmoid', 'linear', 'selu', 'serialize', 
                                 'softmax', 'softplus', 'softsign', 'swish'])
        self.grid_layout.addWidget(self.combo_box, self.n + 1, 3)
        self.layer_all1.append(self.combo_box)
        
          # Increment self.n in each iteration of the loop
        # Create a delete button for the current layer

       # self.mm = self.n - 1
        self.n += 1
   
        
        self.delete_button.setEnabled(True)
    

    def delete_layer(self, n):
        if len(self.node_all1) > 0:
            # Remove the last layer elements associated with the last row from the GUI
            last_label = self.label_all1.pop()  # Remove and get the last label
            self.grid_layout.removeWidget(last_label)  # Remove the label from the layout
            last_label.deleteLater()  # Delete the QLabel
            
            last_s = self.node_all1.pop()  # Remove and get the last QLineEdit
            self.grid_layout.removeWidget(last_s)  # Remove the QLineEdit from the layout
            last_s.deleteLater()  # Delete the QLineEdit
            
            last_combo_box = self.layer_all1.pop()  # Remove and get the last QComboBox
            self.grid_layout.removeWidget(last_combo_box)  # Remove the QComboBox from the layout
            last_combo_box.deleteLater()  # Delete the QComboBox
            
#            del self.label_all1[-1]
#            del self.node_all1[-1]
#            del self.layer_all1[-1]
            
            # Adjust the layout to fill the gap left by the deleted layer
            self.grid_layout.removeItem(self.grid_layout.itemAtPosition(self.n, 0))
            self.grid_layout.removeItem(self.grid_layout.itemAtPosition(self.n, 1))
            self.grid_layout.removeItem(self.grid_layout.itemAtPosition(self.n, 3))
            self.grid_layout.removeItem(self.grid_layout.itemAtPosition(self.n, 4))
            
            # Reduce the count of layers (self.n) by 1
            self.n -= 1
    
            # If there are no more layers, disable the delete button
        if len(self.node_all1) == 0:
                self.delete_button.setEnabled(False)

    

    
    def run_svr2(self):
        try:
                                self.node_all =[]
                                self.layer_all =[]
                                for i in range(len(self.node_all1)):
                                    self.node_all.append(self.node_all1[i].text())
                                    self.layer_all.append(self.layer_all1[i].currentText())
                               
                             #   csv_file_path = "data1.csv"  # Replace with your CSV file path
                             #   data = pd.read_csv(csv_file_path)
                                # canonical training loop for a Pytorch Geometric GNN model gnn_model
                             #   x_smiles = data["SMILES"].tolist()
                               # y_max = max(data["CCS"])
                               # data["CCS"] = data["CCS"]  / y_max
                               # y = data["CCS"].tolist()
                                index1, index, target, df, X, y, df1, X_test1 = data_xy.input0()
                                data = df
                                y_max = max(y)
                                y = (y.reshape(-1,1)/y_max).tolist()
                                try:
                                    x_smiles = df["SMILES"].tolist()
                                    X_test1 = df1["SMILES"].tolist()
                                except Exception as e:
                                    error(e) 
                                
                                bs = int(self.line_edit_1.text())
                                # create list of molecular graph objects from list of SMILES x_smiles and list of labels y
                                data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y)
                                data_list_p = create_pytorch_geometric_graph_data_list_from_smiles_and_labels_p(X_test1)
                                # create dataloader for training
                                dataloader = DataLoader(dataset = data_list, batch_size = bs, collate_fn=custom_collate)
                                dataloader_p = DataLoader(dataset = data_list_p, batch_size = bs, collate_fn=custom_collate)

                                

                                # Assuming data_list is your list of Data objects
                                max_nodes = max(data.x.size(0) for data in data_list)
                                max_features = max(data.x.size(1) for data in data_list)
                                




                                # Preprocess data_list to ensure all feature matrices have the same size
                                for data in data_list:
                                    # Pad or truncate nodes and features to match the maximum size
                                    data.x = torch.cat([data.x, torch.zeros(max_nodes - data.x.size(0), max_features)], dim=0)[:max_nodes]

                                # Split the data into training and testing sets
                      #          data_train, data_test = train_test_split(data_list, test_size=0.2, random_state=42)

                                # Now you can safely use torch.stack to create tensors for X_train and X_test
                        #        X_train = torch.stack([data.x for data in data_train])
                        #        X_test = torch.stack([data.x for data in data_test])


                                # Split your dataset into training, validation, and test sets
                                # Extract X and y from your data_list
                                X = [data.x for data in data_list]
                                y = [data.y for data in data_list]
                                
                                for data in data_list_p:
                                    # Pad or truncate nodes and features to match the maximum size
                                    data.x = torch.cat([data.x, torch.zeros(max_nodes - data.x.size(0), max_features)], dim=0)[:max_nodes]
                                    
                                X_pred = [data.x for data in data_list_p]
                                edge_indices = [data.edge_index for data in data_list]
                                edge_feat = [data.edge_attr for data in data_list]

                             #   
                                
                                edge_index_p = [data.edge_index for data in data_list_p]
                                edge_feat_p = [data.edge_attr for data in data_list_p]                                


                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

                                # Split edge_index (assuming you have a list of edge indices)
                                edge_index_train, edge_index_test, edge_attr_train, edge_attr_test = train_test_split(edge_indices, edge_feat, test_size=0.2, random_state=42)
                                edge_index_train, edge_index_val, edge_attr_train, edge_attr_val = train_test_split(edge_index_train, edge_attr_train, test_size=0.2, random_state=42)
                                
                              #  print(len(X_pred[1]), len(X_pred[2]))
                              #  print(len(X_test[1]), len(X_test[2]))

                                # Define your GNN model
                                input_dim = 79  # Adjust this to match your input feature dimension
                                hidden_dim = self.NL
                                output_dim = 1
                                def GNNModel(input_dim, hidden_dim, output_dim,layer_all, optimizers):
                                    conv1 = GCNConv(input_dim, hidden_dim)      
                                    # Hidden layers
                                    hidden_layers = nn.ModuleList([
                                                GCNConv(hidden_dim, hidden_dim) for _ in range(len(self.layer_all))
                                            ])
                                            
                                    # Output layer
                                    fc1 = nn.Linear(hidden_dim, output_dim)
                                            
                                    # Define activation functions for each layer
                                    
                                    def get_activation(activation):
                                        if activation == 'relu':
                                            return nn.ReLU()
                                        elif activation == 'sigmoid':
                                            return nn.Sigmoid()
                                        elif activation == 'tanh':
                                            return nn.Tanh()
                                        elif activation == 'softmax':
                                            return nn.Softmax(dim=1)
                                        elif activation == 'elu':
                                            return nn.ELU()
                                        elif activation == 'exponential':
                                            return nn.ELU()
                                        # Add other activation functions as needed     
                                        
                                        
                                    activations = nn.ModuleList([
                                                get_activation(activation) for activation in self.layer_all
                                            ])
                                  #  print(self.layer_all)   
                                           # Define the optimizer
                                 #   self.optimizer = optimizer
                                            
   
                                    class GNN(nn.Module):
                                        def __init__(self, input_dim, hidden_dim, output_dim):
                                            super(GNN, self).__init__()
                                            self.conv1 = conv1
                                            self.hidden_layers = hidden_layers
                                            self.fc1 = nn.Linear(hidden_dim, 1)  #fc1
                                            self.activations = activations
                                
                                        def forward(self, x, edge_index, y):
                                            # Apply GNN operations to x and edge_index with input activation
                                            x = self.conv1(x, edge_index)
                                            x = self.activations[0](x)
                                            
                                            # Apply hidden layers
                                            for i, layer in enumerate(self.hidden_layers):
                                                x = layer(x, edge_index)
                                                x = self.activations[i](x)
                                    
                                            # Apply output layer
                                            y_pred = self.fc1(x)
                                
                                            if y is not None:
                                                # Compute the Mean Squared Error (MSE) loss
                                                loss = F.mse_loss(y_pred, y)
                                                return y_pred, loss
                                            else:
                                                return y_pred
                                
                                    # Create an instance of the GNN model
                                    gnn_model = GNN(input_dim, hidden_dim, output_dim)
                                    
                                    
                                           # Map the selected optimizer string to the corresponding optimizer class
                                    optimizer_mapping = {
                                               'adam': optim.Adam,
                                               'SGD': optim.SGD,
                                               'RMSprop': optim.RMSprop,
                                               'Adadelta': optim.Adadelta,
                                               'Adagrad': optim.Adagrad,
                                               'Adamax': optim.Adamax,
                                               'AdamW': optim.AdamW,
                                               'SparseAdam': optim.SparseAdam,
                                               'LBFGS': optim.LBFGS
                                             #  'Nadam': optim.Nadam,
                                             #  'Ftrl': optim.Ftrl,
                                           }
                                           
                                           # Use the selected optimizer class to create an optimizer instance
                                    optimizer_class = optimizer_mapping[optimizers]
                                    learning_rate = 0.001 
                                    optimizer_instances = optimizer_class(gnn_model.parameters(), lr=learning_rate)  # Replace 'learning_rate' with your desired value
                                    

                                    # Create a list of optimizer instances based on the optimizer strings
                                 #   optimizer_instances = [optim.Adam(gnn_model.parameters(), lr=lr) for optimizer, lr in optimizers]
                                    
                                    return gnn_model, optimizer_instances
                                
                                
                                optimizers=self.combo_box1.currentText()
                                gnn_model, optimizers_list = GNNModel(input_dim, hidden_dim, output_dim, self.layer_all, optimizers)
                                # Now you can use 'optimizer_instance' in your GNNModel function
                                       # Create a list of optimizers for each layer
                                
                                criterion = torch.nn.MSELoss()
                                # Define optimizer and loss function
                                optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
                                criterion = torch.nn.MSELoss()



                                
                                # Create a custom dataset
                                custom_dataset = CustomGraphDataset(X_train, edge_index_train, edge_attr_train, y_train)

                                # Create a DataLoader for training
                                train_loader = DataLoader(custom_dataset, batch_size=bs, shuffle=True,collate_fn=custom_collate)
                                test_dataset = CustomGraphDataset(X_test, edge_index_test,edge_attr_test, y_test)
                                test_loader = DataLoader(test_dataset, batch_size=bs,collate_fn=custom_collate)


                                val_dataset = CustomGraphDataset(X_val, edge_index_val,edge_attr_val, y_val)
                                val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True, collate_fn=custom_collate)
                                
                                pred_dataset = CustomGraphDataset_p(X_pred, edge_index_p,edge_feat_p)
                                pred_loader = DataLoader(pred_dataset, batch_size=1,shuffle=True, collate_fn=custom_collate)

                                num_epochs = int(self.line_edit1.text())

                                # Training loop
                                for epoch in range(num_epochs):
                                    gnn_model.train()  # Set the model to training mode
                                    total_loss = 0

                                    for batch_data in train_loader:
                                        optimizer.zero_grad()
                                        x, edge_index,edge_attr, y = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.y
                                        
                                        predictions = gnn_model(x, edge_index,None) 
                                        predictions = torch.mean(predictions).view(-1) #F.softmax(predictions, dim = 1)
                                     #   print(predictions.shape)     
                                        loss = F.mse_loss(predictions, y.view(-1))
                                      #  print(predictions.detach().numpy()[0])
                                        loss.backward()
                                        optimizer.step()
                                        total_loss += loss.item()
                                    avg_loss = total_loss / len(train_loader)
                                    print(f'Epoch [{epoch+1}/{num_epochs}] - Avg. Loss: {avg_loss:.4f}')
                                  #  print(predictions*y_max)

                                # Optionally, save your trained model
                                torch.save(gnn_model.state_dict(), 'gnn_model.pth')

                                gnn_model.eval()  # Set the model to evaluation mode
                                total_val_loss = 0


                                with torch.no_grad():
                                    for batch_data in val_loader:
                                        x, edge_index, y = batch_data.x, batch_data.edge_index, batch_data.y
                                        predictions = gnn_model(x, edge_index, None)
                                        predictions = torch.mean(predictions).view(-1)
                                        loss = F.mse_loss(predictions, y.view(-1))
                                   #     loss.backward()
                                    #    print(loss)
                                     #   optimizer.step()
                                        total_val_loss += loss.item()
                                # Calculate average validation loss
                                avg_val_loss = total_val_loss / len(val_loader)
                                print(f"Validation Loss: {avg_val_loss:.4f}")
                                #print([x * y_max for x in y_val], predictions*y_max)


                                # Optionally, calculate and print other evaluation metrics

                                gnn_model.eval()  # Set the model to evaluation mode
                                total_test_loss = 0
                                test_predictions = []  # Store predicted values
                                ty = []
                                with torch.no_grad():
                                    for batch_data in test_loader:
                                        x, edge_index, y = batch_data.x, batch_data.edge_index, batch_data.y
                                        predictions = gnn_model(x, edge_index, None)
                                        predictions = torch.mean(predictions).view(-1)
                                       # print(predictions.shape, y.view(-1).shape)
                                        loss = F.mse_loss(predictions, y.view(-1))
                                        # Append the predicted values to the list
                                        test_predictions.append(predictions.detach().numpy())
                                        ty.append(y)
                                        # Compute the Mean Squared Error (MSE) loss
                                       # loss = torch.nn.MSELoss()(predictions, y)
                                        total_test_loss += loss.item()                               
                                # Calculate average test loss
                                avg_test_loss = total_test_loss / len(test_loader)
                                print(f"Test Loss: {avg_test_loss:.4f}")
                                # Convert the test_predictions list to a NumPy array if needed
                                test_predictions = np.array(test_predictions)

                                # Now you can print the test_predictions and corresponding ground truth (y values)
                      #          print("Predicted Values:", test_predictions)
                      #          print("Ground Truth (y values):", y.cpu().numpy())  # Assuming 'y' contains your target values
                                
                                numpy_array_list = [tensor.numpy() for tensor in ty]
                                y_test_1 = [arr[0, 0] for arr in numpy_array_list]
                                   
                                true_labels = [i*y_max for i in y_test_1]
                               #predictions = model.predict(test_data.batch(1), verbose=0)[:, 0]
                                predictions_1 = [i*y_max for i in test_predictions]
                                predictions_test = [arr[0] for arr in predictions_1]
                                
                                APE=100*(abs(np.array(true_labels)-np.array(predictions_test))/np.array(true_labels))
                        
                                # Defining a function to find the best parameters for ANN


                                gnn_model.eval()  # Set the model to evaluation mode
                                total_pred_loss = 0
                                pred_predictions = []  # Store predicted values
                                with torch.no_grad():
                                    for batch_data_p in pred_loader:
                                        x_p, edge_index_pred = batch_data_p.x, batch_data_p.edge_index
                                        predictions_p = gnn_model(x_p, edge_index_pred, None)
                                        # Append the predicted values to the list
                                        pred_predictions.append(predictions_p.detach().numpy())
                                
                        #        print([len(pred_predictions[i]) for i in range(len(pred_predictions))])
                                pred_predictions = np.array(pred_predictions)
                                Predictions1 = [i*y_max for i in pred_predictions]
                                
                                
                                dialog = QtWidgets.QDialog()
                                dialog.setWindowTitle('GNN: ' + str(self.NL) +  ' layers results')
                                MyIcon(dialog)
                                layout = QtWidgets.QVBoxLayout(dialog)

                                text_widget = QtWidgets.QLabel()
                                figure_widget = QtWidgets.QWidget()
                                        
                                layout.addWidget(text_widget)
                                layout.addWidget(figure_widget)
                                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
                            #    ax.plot(true_labels ,true_labels)
                             #   ax.legend()
                              #  ax.xlabel("Epoch")
                              #  ax.ylabel("Loss")
                              #  ax.show()
                                ax.scatter(true_labels, predictions_test)
                                ax.plot(true_labels, true_labels, '-g', linewidth=2)
                                ax.set_xlabel("True_label")
                                ax.set_ylabel("Prediction")

                                
                                canvas1 = FigureCanvas(fig)
                                layout.addWidget(canvas1)
                              #  canvas1.destroyed.connect(fig.clf)
                                
                                
                                
                                
                                
                                text_widget.setText("Features: \n")
                                layout.addWidget(text_widget)

                 
                                for x in index1:
                                    text_widget.setText(text_widget.text()+' '.join(x)+'\n')
                              
                                text_widget.setText(text_widget.text()+  '========================= '+ '\n')  
                                text_widget.setText(text_widget.text()+  'Number of nodes: '+ '\n')
                                i = 0
                                for x in self.node_all:
                                    if i == 0:
                                        text_widget.setText(text_widget.text()+ 'Input Layer: ' +' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                    elif i == len(self.node_all)-1 :
                                        text_widget.setText(text_widget.text()+ 'Output Layer: ' +  ' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                        
                                    else:
                                        text_widget.setText(text_widget.text()+ 'Hidden Layer #' + str(i+1) + ' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                    i += 1
                                    
                                text_widget.setText(text_widget.text()+ '========================= '+ '\n')      
                                text_widget.setText(text_widget.text()+  'The Average Loss of  model is: ' + str(avg_loss) + '\n')    
                          #      text_widget.setText(text_widget.text()+  'RMSE = ' + str(sqrt(mean_squared_error(y_test_orig[:,0],Predictions[:,0]))))   
                                text_widget.setText(text_widget.text()+ '========================= '+ '\n')  
                                text_widget.setText(text_widget.text()+ 'The Accuracy of  Deep GNN model model is (Test data): ' + str(100-np.mean(APE)) + '\n')
       
                                dialog.exec ()

                                
                                textfile = open(self.pixmappath  + 'Output.txt', 'w')
                             #   textfile.write('Total time = ' + str( round((EndTime-StartTime)/60))+ ' Minutes'+ '\n')   
                             #   textfile.write('========================= '+ '\n')  
                                textfile.write('Features: '+ '\n')
                        
                                for x in index1:
                                    textfile.write(x + '\n')
                                textfile.write('========================= '+ '\n')  
                                textfile.write('Number of nodes: '+ '\n')
                                i = 0
                                for x in self.node_all:
                                    if i == 0:
                                        textfile.write('Input Layer: ' +' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                    elif i == len(self.node_all)-1 :
                                        textfile.write('Output Layer: ' +  ' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                        
                                    else:
                                        textfile.write('Hidden Layer #' + str(i+1) + ' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                    i += 1
                                    
                                textfile.write('========================= '+ '\n')
                                textfile.write('The Accuracy of  Deep GNN model model is: ' + str(100-np.mean(APE)) + '\n')
                                textfile.write('RMSE = ' + str(sqrt(mean_squared_error(true_labels,predictions_test)))+ '\n')  
                                textfile.write('y_test ' + 'y_predict'+ '\n') 
                                data = np.column_stack([true_labels,predictions_test])
                                np.savetxt(textfile , data, fmt=['%d','%-4d'])
                                textfile.close()     #   self.root.mainloop()
                               # print(len(df1), len())
                                
                                textfile2 = open(self.pixmappath + 'GNN_'+'Output_test.txt', 'w')
                                df1[target+'_prediction']=Predictions1
                                df1.to_csv(self.pixmappath + 'GNN_Output_test.csv', index=False, lineterminator='\n')
                                textfile2.close()
                                return
        except Exception as e:
            error(e) 
                            
def GNN2():
 #   app = QApplication(sys.argv)
    window = GNN()
    window.show()
    window.exec_()   

#GNN2()    