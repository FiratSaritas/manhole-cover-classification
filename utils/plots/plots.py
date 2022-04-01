from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import to_pil_image

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
        
    @classmethod
    def loss_sma(self, loss, sma=50, show_fig = True):
        colors = sns.color_palette('Paired', 4)
        sns.set_style('white')

        mean_loss_folds = loss.rolling(sma).mean()
        std_loss_folds = loss.rolling(sma).std()

        p = sns.lineplot(x=mean_loss_folds.index, y=mean_loss_folds, label='Mean Batch', color=colors[1])
        p = sns.lineplot(x=mean_loss_folds.index, y=mean_loss_folds + std_loss_folds, 
                         label=r'$\pm1\sigma$', color=colors[0], linestyle='--', alpha=.5)
        p = sns.lineplot(x=mean_loss_folds.index, y=mean_loss_folds - std_loss_folds, 
                         color=colors[0], linestyle='--', alpha=.5)
        plt.text(x=mean_loss_folds.index[-1], y=mean_loss_folds.iloc[-1], 
                 s=str(round(mean_loss_folds.iloc[-1], 2)), va='center')

        p.set_title(f'Loss over Batches / SMA{50}',loc='left')
        p.set_xlabel('Batches')
        p.set_ylabel('Cross-Entropy Loss')
        sns.despine()
        if show_fig:
            plt.show()
        
    @classmethod 
    def visualize_model(self, model, dataloader, label_translator, num_images=6, ncol=2):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                       std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                      std = [ 1., 1., 1. ]),
                           ])
        was_training = model.training
        model.eval()
        images_so_far = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                fig = plt.subplots(figsize=(18, (num_images//ncol) * 5))
                for j in range(num_images):
                    plt.subplot(num_images//ncol+1, ncol, j+1)
                    plt.title('predicted: {} / actual: {}'.format(label_translator[preds[j].item()],
                                                                  label_translator[labels[j].item()]))
                    img = inputs.cpu().data[j]
                    img = invTrans(img)
                    img = to_pil_image(img)
                    plt.axis('off')
                    plt.imshow(img)
                    images_so_far += 1

                model.train(mode=was_training)
                plt.subplots_adjust(hspace=.3)
                plt.show()
                break


        
    
    