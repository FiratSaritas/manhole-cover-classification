from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
sns.set_palette('Paired')



class EvaluationPlots:
    
    @classmethod
    def plot_train_val_loss(self, train_loss, val_loss, **kwargs):
        fig = plt.subplots(figsize=(14, 4))
        plt.subplot(1,2,1)
        p = sns.lineplot(x=np.arange(len(train_loss)), 
                         y=train_loss, label='Batch Loss')
        p.set_title('Training Loss', loc='left')
        p.set_xlabel('Batches')
        p.set_ylabel('Loss')
        sns.despine()
        
        plt.subplot(1,2,2)
        p = sns.lineplot(x=np.arange(len(val_loss)), y=val_loss)
        p.set_title('Validation Loss', loc='left')
        p.set_xlabel('Batches')
        p.set_ylabel('Loss')
        sns.despine()

        plt.show()
        
    @classmethod
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_labels: dict):
               
        plt.subplots(figsize=(10, 8))
        p = sns.heatmap(confusion_matrix(y_true=y_true, y_pred=y_pred), cmap='Blues',
                        xticklabels=class_labels.values(), yticklabels=class_labels.values(), 
                        square=True, annot=True, fmt="d")

        p.set_title('Confusion Matrix', loc='left')
        p.set_xlabel('Predicted', rotation=20, ha='right')
        p.set_ylabel('True')

        plt.show()

    
        
    
    