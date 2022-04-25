from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torchvision import transforms as transforms
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import to_pil_image

sns.set_palette('Paired')




class EvaluationPlots:
    """
    This function is used to plot different visualizations of the model
    """
    
    @classmethod
    def plot_train_val_loss(self, train_loss, val_loss, **kwargs):
        """plot train and validation loss

        Args:
            train_loss (_type_): _description_
            val_loss (_type_): _description_
        """

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
        """plot confusion matrix

        Args:
            y_true (np.ndarray): _description_
            y_pred (np.ndarray): _description_
            class_labels (dict): _description_
        """
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
        """plot loss with moving average

        Args:
            loss (_type_): _description_
            sma (int, optional): _description_. Defaults to 50.
            show_fig (bool, optional): _description_. Defaults to True.
        """
        colors = sns.color_palette('Paired', 4)
        sns.set_style('white')
        if not isinstance(loss, pd.Series):
            loss = pd.Series(loss)

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
        """visualize model predictions and actual labels

        Args:
            model (_type_): _description_
            dataloader (_type_): _description_
            label_translator (_type_): _description_
            num_images (int, optional): _description_. Defaults to 6.
            ncol (int, optional): _description_. Defaults to 2.
        """
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

    def plot_pred(y_true, y_pred, y_prob, y_images,n_images,label_dict, incorrect = True):
        """plot images with their predictions and probabilities

        Args:
            y_true (_type_): _description_
            y_pred (_type_): _description_
            y_prob (_type_): _description_
            y_images (_type_): _description_
            n_images (_type_): _description_
            label_dict (_type_): _description_
            incorrect (bool, optional): _description_. Defaults to True.
        """
        corrects = y_true == y_pred
        example = []
        for image, label, prob, correct in zip(y_images, y_true, y_prob, corrects):
            if incorrect == True:
                if not correct:
                    example.append((image, label, prob))
            elif incorrect == False:
                if correct:
                    example.append((image, label, prob))
        rows = int(np.sqrt(n_images))
        cols = int(np.sqrt(n_images))

        fig = plt.figure(figsize=(10, 10))
        for i in range(rows*cols):
            ax = fig.add_subplot(rows, cols, i+1)
            image, true_label, probs = example[i]
            true_prob = probs[true_label]
            example_prob, example_label = torch.max(probs, dim=0)
            true_label = list(label_dict.keys())[list(label_dict.values()).index(true_label)]
            example_label = list(label_dict.keys())[list(label_dict.values()).index(example_label)]
            ax.imshow(image.T)
            ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n'
                         f'pred label: {example_label} ({example_prob:.3f})')
            ax.axis('off')
        fig.subplots_adjust(hspace=0.5)
        
    def plot_metrics(metrics):
        """Bar plot of metrics

        Args:
            metrics (_type_): _description_
        """
        df_metrics = pd.DataFrame.from_dict(metrics)
        ax = df_metrics.plot(kind='bar', figsize=(6, 4), rot=0, title='Metrics Comparison', ylabel='Values')
        ax.legend(bbox_to_anchor=(0.5,-0.1))
        plt.show()
        
        
    def metric_evaluation(y_test, y_pred,classes):
        """Plots the confusion matrix and accuracy score

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            classes (_type_): _description_
        """

        conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

        fig = plt.subplots(figsize=(20, 5))
        plt.subplot(1,2,1)
        p1 = sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=classes,yticklabels=classes, cmap="YlGnBu")
        p1.set_title('Confusion Matrix')
        p1.set_ylabel('True')
        p1.set_xlabel('Predicted')

        plt.subplot(1,2,2)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        metric_names = ['Recall Macro', 'Precision Macro', 'F1 Macro']
        metrics = [recall, precision, f1]

        p2 = sns.barplot(x=metric_names, y=metrics,palette="Blues_d")
        for i, value in enumerate(metrics):
            plt.text(x=i, y=value, s=str(round(value,3)), ha='center')
        p2.set_ylim(0,1) 
        p2.set_title('metrics')
        p2.set_ylabel('Score')
        sns.despine()

        plt.show()

    