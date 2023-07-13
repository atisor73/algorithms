import numpy as np

def map_palette(q, palette, nan_color='#000000'):
    """
    Arguments
    ---------------
    q: 1D quantitative axis, numpy array
    palette: color palette to map to
    
    Returns
    ---------------
    colors: list of mapped colors
    """
    if type(q) is list:
        q = np.array(q)
    q_min = np.nanmin(q)
    q_max = np.nanmax(q)
    
    indices = ((q - q_min)/(q_max - q_min) \
               * (len(palette)-1)).astype(int)
    
    colors = []
    for i in indices:
        if np.isnan(i):
            color = nan_color
        else:
            color = palette[i]
        colors.append(color)
        
    
    return colors




def map_palette_thresh(q, palette, max_thresh='inf', max_color=None,
                       min_thresh='-inf', min_color=None, nan_color='#000000'):
    """
    Useful for creating legends.
    
    Arguments
    ---------------
    q: 1D quantitative axis, numpy array
    palette: color palette to map to

    Returns
    ---------------
    colors: list of mapped colors
    """
    
    if max_color is None:
        max_color = palette[-1]
    if min_color is None:
        min_color = palette[0]
        
    if type(q) is list:
        q = np.array(q)
        
    if max_thresh == 'inf':
        max_thresh = np.nanmax(q)
    if min_thresh == '-inf':
        min_thresh = np.nanmin(q)
    
    colors = []
    for v in q:
        if np.isnan(v):
            c = nan_color
        elif v >= max_thresh: c = max_color
        elif v <= min_thresh: c = min_color
        else:
            i = ((v - min_thresh)/(max_thresh - min_thresh) \
               * (len(palette)-1))
            c = palette[int(i)]
            
        colors.append(c)
    return colors
