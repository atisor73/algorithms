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
