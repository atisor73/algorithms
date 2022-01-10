import bokeh.io
import bokeh.plotting
import numpy as np
import pandas as pd
import colorcet

def _overlap(df, i,   thresh, size, tssq):
    '''return prev overlaps'''
    xcurrent, ycurrent = df.iloc[i]['x'], df.iloc[i]['y']
    prev = df[:i]
    where = ( (xcurrent-prev['x'] < 2*size) & ((tssq- (xcurrent-prev['x'])**2) > ((ycurrent-prev['y'])**2)) )
    return where

sgn = 1                                         # global toggle
def _jitterY(df, i, layers,   thresh, size, tssq):
    global sgn
    xcurrent, ycurrent = df.iloc[i]['x'], df.iloc[i]['y']
    where = _overlap(df, i, thresh, size, tssq)       # all previous overlaps
    prev = df[:i]

    _ = 0
    for __ in range(layers):
        if _ == 0: _, sgn = _+1, sgn*-1         # change incr direction once for each point
        prev_overlap = prev[where]              # subset of `prev` from `overlap` function

        yoff=0
        for xprev, yprev in zip(prev_overlap.x, prev_overlap.y):
            if sgn ==1: ynew = np.sqrt(tssq - (xcurrent-xprev)**2) + yprev     # deterministically calc.
            elif sgn==-1: ynew = yprev - np.sqrt(tssq - (xcurrent-xprev)**2)
            _yoff = np.abs(ynew-ycurrent)
            if _yoff > yoff:  yoff = _yoff

        ycurrent += sgn*yoff
        df.loc[i,'y'] = ycurrent

        where = _overlap(df, i, thresh, size, tssq)
        if not any(where):
            break
    return df

scale_y = 10
def swarm(arr, width, layers, thresh, size):
    """
    A very fussy approach to getting (x, y) coordinates for a swarm plot.

    Arguments:
    ----------
    arr: axis of quantitative values (iterable)
    width: plot width, used for intermediate scaling purposes (int)
    layers: # of jittering loop, higher = better performance, but much slower (int)
    thresh: scales inter-point distance (float)
    size: point size (float)

    Returns:
    --------
    df: Pandas dataframe of (x, y) coordinates
    """
    # ***************** quant. axis in unit pixel ******************
    x_min, x_max = min(arr), max(arr)
    scale_x = width/(x_max-x_min)
    arr*=scale_x

    x_orig, y_orig = arr, [0]*len(arr)
    y_new = [0]                                # first jitter's 0
    df = pd.DataFrame({'x':arr,'y':y_orig})

    tssq = (thresh*size+size)**2               # only need to compute this once
    for i in range(len(arr)):
        df = _jitterY(df, i, layers, thresh, size, tssq)

    df['x'] /= scale_x
    df['y'] /= scale_y
    return df

def swarmplot(
    arrs,
    size=3,
    thresh=1, # scaling of inter-pt distance
    width=600, height=400,
    x_range=None, y_range=None,
    layers=10,   # layers of comparison
    palette=['#0d936a','#2c2f30','#e3a201','#6762ab','#d55101','#8fa7d7','#afd7d6','#aa3751','#f5b3b8']+colorcet.glasbey,
    title=None,
    x_axis_label=None
):
    """
    A very janky attempt at a swarm plotter.

    Arguments
    ----------
    arrs: iterable along categorical axis of quantitative values ([[], [], ...])
    size: point size (float)
    thresh: scales inter-point distance (float)
    layers: # of jittering loop, higher = better performance, but much slower (int)
    palette: categorical axis palette (list)
    width: plot width (int)
    height: plot height (int)
    title: plot title (str)

    Returns:
    --------
    p: Bokeh plotting figure
    """
    if palette == None:
        palette=['#0d936a','#2c2f30','#e3a201','#6762ab','#d55101','#8fa7d7','#afd7d6','#aa3751','#f5b3b8']+colorcet.glasbey
    p=bokeh.plotting.figure(
        width=width,height=height,title=title,
        x_range=x_range,y_range=y_range, x_axis_label=x_axis_label)
    off = 0
    for _, arr in enumerate(arrs):
        arr = np.sort(arr)
        _df = swarm(arr, width, layers, thresh, size)
        _df['y'] += off
        p.circle(x=_df['x'], y=_df['y'], color=palette[_], size=size)

        interswarm_dist = (_df['y'].max()-_df['y'].min())*1.4
        off += interswarm_dist
    return p
