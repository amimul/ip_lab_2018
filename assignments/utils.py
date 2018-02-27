#coding=utf-8
import cv2
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
    
