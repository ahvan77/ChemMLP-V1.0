import sys
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QDialog,QGridLayout, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel
from icon import MyIcon
from Err import error
import traceback
import ml_GUI_de
import ANNew
import GNN_chem
import LR
import PLS
import GBR
import RFR
import KNR
import VR
import LASSO
#import QML
#import Chemprop

class MyDialog(QDialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)
        MyIcon(self)
        self.setWindowTitle("ML Models")
        # Create combo box to select model type
        self.combo_box = QComboBox()
        self.combo_box.setMinimumSize(200, 30)
        self.combo_box.addItems(['Regression Models', 'Deep Learning', 'GNN (Molecoule)', 'Quantum Machine Learning'])
        
        # Create submit button
        self.submit_button = QPushButton('Submit')
        self.submit_button.clicked.connect(self.on_submit)
        
        # Create layout and add widgets
        layout = QGridLayout()
        label = QLabel('Select the Model:')
        font = QFont('Times New Roman', 10)
        label.setFont(font)
        layout.addWidget(self.combo_box, 0, 1)
        layout.addWidget(self.submit_button, 1, 1)
        
        # Set layout
        self.setLayout(layout)
    


    def on_submit(self):
        selected_model = self.combo_box.currentText()
        def error(e):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle('Error')
            msg.setText('An error has occurred:\n\n{}'.format(str(e)))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
        def my_upd2(combo_box): 
                
                if combo_box.currentText() == 'SVR':
                    try:
                        ml_GUI_de.SVR2()
                    except Exception as e:
                        error(e)   
                elif combo_box.currentText() == 'LASSO':
                    try:
                        LASSO.Las()   
                    except Exception as e:
                        error(e)    
                elif combo_box.currentText() == 'LR':
                    try:
                        LR.LR()   
                    except Exception as e:
                        error(e)    
                elif combo_box.currentText() == 'PLS':
                    try:
                        PLS.PLS() 
                    except Exception as e:
                        error(e)    
                elif combo_box.currentText() == 'GBR':
                    try:
                        GBR.GBR()
                    except Exception as e:
                        error(e)                        
                elif combo_box.currentText() == 'RFR':
                    try:
                        RFR.RFR()   
                    except Exception as e:
                        error(e)                        
                elif combo_box.currentText() == 'KNR':
                    try:
                        KNR.KNR()  
                    except Exception as e:
                        error(e)                        
                elif combo_box.currentText() == 'VR':
                    try:
                        VR.VR()   
                    except Exception as e:
                        error(e)                        
        if selected_model == 'Regression Models':
            dialog = QDialog()
            dialog.setWindowTitle('Regression Models')
            MyIcon(dialog)
         #   dialog.setGeometry(200, 200, 50, 50)
        #    dialog.setGeometry(100, 100, 400, 100)
        
            label = QLabel('Select the ML Model:')
            font = QFont('Times New Roman', 10)
            label.setFont(font)
        
            combo_box = QComboBox()
          #  self.combo_box.setMinimumSize(200, 30)
            combo_box.addItems(['SVR', 'LASSO', 'LR', 'PLS', 'GBR', 'RFR', 'KNR', 'VR'])
            combo_box.setCurrentIndex(0)
        
            submit_button = QPushButton('Submit')
            submit_button.clicked.connect(lambda: my_upd2(combo_box))
        
            layout = QVBoxLayout()
            layout.addWidget(label)
            layout.addWidget(combo_box)
            layout.addWidget(submit_button)
            dialog.setLayout(layout)
           # dialog.setWindowModality(Qt.ApplicationModal)
            
         #   dialog.setWindowTitle('Deep Learning')
            dialog.show()
            dialog.exec_()
         #   dialog.close()
            # Do something for Regression Models
        elif selected_model == 'Deep Learning':
            # Import and run ANN_GUI_de.ANN_2L() for Deep Learning
            try:
                ANNew.ANN()
            except Exception as e:
                error(e)  
        elif selected_model == 'GNN (Molecoule)':
            try:
                GNN_chem.GNN2()
            except Exception as e:
                error(e)  
            
    #    elif selected_model == 'Quantum Machine Learning':
            # Import and run ANN_GUI_de.ANN_2L() for Deep Learning
     #       try:
                #Chemprop()
      #          QML.Qmodels()
      #      except Exception as e:
       #         error(e)                  
def models():
#    app = QApplication([]) 
    dialog = MyDialog()
    dialog.show()
    dialog.exec_()
   # dialog.close()
 #   sys.exit(app.exec_())
    

