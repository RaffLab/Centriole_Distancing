#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains scripts for visualising detections.

"""
import numpy as np 
import pylab as plt 

def draw_circles(points, ax, radii=3, col='r', lw=2, to_fill=False):
    """ Given (y,x) centroid coordinates, draws on a given Matplotlib ax, circles of a defined radius and colour.

    Parameters
    ----------
    points : numpy array or list
        an array of (y,x) coordinates for plotting
    ax : Matplotlib figure axes instance
        e.g. from fig, ax = plt.subplots()
    radii : int
        the radius of the circle
    col : string
        a valid Matplotlib colour string or valid RGBA float values
    lw : int 
        linewidth of circle
    to_fill : bool
        if True, fills the circle with same colour as given by `col`.

    Returns
    -------
    Nothing 

    """
    for p in points:
        y, x = p
        c = plt.Circle((x, y), radii, color=col, linewidth=lw, fill=to_fill)
        ax.add_patch(c)
        
    return []


def draw_squares(points, ax, width, col, lw=2):
    """ Given (y,x) centroid coordinates, draws on a given Matplotlib ax, squares of given width.

    Parameters
    ----------
    points : numpy array or list
        an array of (y,x) coordinates for plotting
    ax : Matplotlib figure axes instance
        e.g. from fig, ax = plt.subplots()
    width : int
        width of the square
    col : string
        a valid Matplotlib colour string or valid RGBA float values
    lw : int 
        line thickness of square
    
    Returns
    -------
    Nothing 

    """
    for p in points:
        y,x = p
        x1 = x - width/2.
        x2 = x1+ width
        y1 = y-width/2.
        y2 = y1 + width
        ax.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], color=col, lw=lw)
        
    return []