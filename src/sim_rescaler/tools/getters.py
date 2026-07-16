import numpy as np
import ROOT

CHAN_LIST = ('PAS1','PBS1','PCS1','PDS1','PES1','PFS1',
             'PAS2','PBS2','PCS2','PDS2','PES2','PFS2')

def get_hist_params(arr,widen=True):
    if len(arr) == 0:
        xlow = -1e-12
        xup = 1e-12
        nbins = 1

        old_xlow = xlow
        old_xup = xup
        old_nbins = nbins
    else:
        old_xlow = np.min(arr)
        old_xup = np.max(arr)
        old_nbins = int(len(arr) ** 0.5)
        
        bin_width = (old_xup - old_xlow) / old_nbins

        widen_by = 0.3 * (old_xup - old_xlow)
        xlow = old_xlow - widen_by
        xup = old_xup + widen_by
    
        nbins = get_nbins(xlow,xup,bin_width)

    if widen:
        return xlow,xup,nbins
    else:
        return old_xlow,old_xup,old_nbins

def get_hist(arr,name,title=None,xlow=None,xup=None,nbins=None,widen=True,normalize=False):
    if title is None:
        title = name

    other_xlow,other_xup,other_nbins = get_hist_params(arr,widen)
    if xlow is None:
        xlow = other_xlow
    if xup is None:
        xup = other_xup
    if nbins is None:
        nbins = other_nbins
    
    hist = ROOT.TH1D(name,title,nbinsx=nbins,xlow=xlow,xup=xup)
    hist[...] = np.histogram(arr,bins=nbins,range=(xlow,xup))[0]
    if normalize:
        # hist.Scale(1.0/hist.Integral())
        hist.Scale(1.0 / hist.GetEntries())
    return hist

def get_nbins(xlow,xup,bin_width):
    if isinstance(bin_width,float):
        return int(np.ceil((xup-xlow)/bin_width))
    else:
        return None

def get_ring_or_central(chan):
    chan = chan.upper()
    if 'PAS' in chan or 'PBS' in chan:
        return "ring"
    elif chan in CHAN_LIST:
        return "central"
    else:
        raise ValueError("Invalid channel")

def get_param_lt(params,n_params):
    params = iter(params)
    return [next(params) for _ in range(n_params)]