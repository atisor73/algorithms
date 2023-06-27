import numpy as np
from IPython.lib.display import Audio

def beep(t=1):
    '''
    Plays sound in jupyter code cell.
    ----------
    t : play time                                (int)
    '''
    frame_rate = 4410
    _t = np.linspace(0, t, frame_rate*t)
    audio_data = np.sin(2*np.pi*300*_t) + np.sin(2*np.pi*240*_t)

    return Audio(audio_data, rate=frame_rate, autoplay=True)
