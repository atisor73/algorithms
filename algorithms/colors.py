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
    q_min = np.nanmin(q)
    q_max = np.nanmax(q)
    
    indices = ((q - q_min)/(q_max - q_min) \
               * (len(palette)-1)).astype(int)
    
    colors = []
    for i in indices:
        try:
            color = palette[i]
        except:
            color = nan_color
        colors.append(color)
        
    
    return colors




def map_palette_thresh(q, palette, max_thresh='inf', max_color="#000000",
                       min_thresh='-inf', min_color="#ffffff"):
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
    if max_thresh == 'inf':
        max_thresh = q.max()
    if min_thresh == '-inf':
        min_thresh = q.min()
    
    colors = []
    for v in q:
        if v > max_thresh: c = max_color
        elif v < min_thresh: c = min_color
        else:
            i = ((v - min_thresh)/(max_thresh - min_thresh) \
               * (len(palette)-1)).astype(int)
            c = palette[i]
        colors.append(c)
    return colors
