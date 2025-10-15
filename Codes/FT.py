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
from sklearn.preprocessing import StandardScaler
import sys
from PySide6.QtWidgets import *
from PySide6 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtGui import QIcon
from icon import MyIcon
from Err import error
import csv
import Predict_methods
import ML_GUI
import os


class F_T(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.init_ui()
        

    def  init_ui(self): 
        self.pixmappath = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
        self.numfeature = []
        self.edit = [] 
        self.n = 0
        
     #   self.setGeometry(100, 100, 50, 50)
        self.setWindowTitle('Features and Target')
        MyIcon(self)
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        
        self.button1 = QPushButton('Insert New Feature', self)
        self.button1.clicked.connect(self.run_svr)
        self.grid_layout.addWidget(self.button1, 0, 0)
        
        
        self.button1 = QPushButton('Insert Data to Predict', self)
        self.button1.clicked.connect(self.run)
        self.grid_layout.addWidget(self.button1, 0, 6)
        
        self.label0 = QLabel('Target: ', self)
        self.grid_layout.addWidget(self.label0, 1,5)
        self.line_edit0 = QLineEdit(self)
        self.grid_layout.addWidget(self.line_edit0, 1, 6)
        
        
        self.predict_checkbox = QCheckBox("Predict data selected")
        self.predict_checkbox.setToolTip("Predict data")
        self.predict_checkbox.setChecked(False)
        self.grid_layout.addWidget(self.predict_checkbox, 0, 3)
        
        

        
    def  run_svr(self):  
        self.n += 1
        s = 'edit_' + str(self.n)
        try:
            self.label1 = QLabel('Feature # ' + str(self.n), self)
            self.grid_layout.addWidget(self.label1, self.n, 0)
            self.s = QLineEdit(self)
            self.grid_layout.addWidget(self.s, self.n, 1)
            self.edit.append(self.s)

        except Exception as e:
            error(e)
        
    def  run(self): 
        Fea = False
        try: 
            for i in range(self.n):
                self.numfeature.append(self.edit[i].text())
                
              
            self.numfeature.append(self.line_edit0.text())  
            entry_list =''
            for entries in self.numfeature:
                    entry_list += str(entries)+'\n'
            textfile = open(self.pixmappath + 'Feature_Target.txt', 'w')
            print(self.pixmappath )
            textfile.write(entry_list)
            textfile.close()
            Fea = True
            if Fea == True:
                if self.predict_checkbox.isChecked():
                   ML_GUI.models() 
                else:
           	       Predict_methods.models()
           	     
             
        except Exception as e:
            error(e)          

def FT():

    #app = QApplication(sys.argv)
    window = F_T()
    window.show()
    window.exec_()   
       