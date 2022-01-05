def map_palette(q, palette):
    """
    Arguments
    ---------------
    q: 1D quantitative axis, numpy array
    palette: color palette to map to 
    
    Returns
    ---------------
    colors: list of mapped colors
    """
    indices = ((q - q.min())/(q.max() - q.min()) \
               * (len(palette)-1)).astype(int)
    colors = [palette[i] for i in indices]
    return colors