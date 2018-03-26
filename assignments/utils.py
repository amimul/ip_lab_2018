#coding=utf-8
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt

def display_image(mat, axes=None, cmap=None):
    """
    Display a given matrix into Jupyter's notebook
    
    :param mat: Matrix to display
    :param axes: Subplot on which to display the image
    :param cmap: Color scheme to use 
    :return: Matplotlib handle
    """
    img = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB) if mat.ndim == 3 else mat
    cmap= cmap if mat.ndim != 2 or cmap is not None else 'gray'
    if axes is None:
        return plt.imshow(img, cmap=cmap)
    else:
        return axes.imshow(img, cmap=cmap)
    
    
def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          axes=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    :param cm: Confusion matrix
    :param classes: Classes name
    :param normalize: Indicate if the confusion matrix need to be normalized
    :param title: Plot's title
    :param axes: Subplot on which to display the image
    :param cmap: Colormap to use
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # Show cm
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #fmt = '.2f' if normalize else 'd'
    #thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    