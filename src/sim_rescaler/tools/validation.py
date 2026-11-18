import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from matplotlib.ticker import FormatStrFormatter
from contextlib import nullcontext

fsize = (17,4)

def frac_plotter(f_sim_x,f_sim_y,f_data_x,f_data_y,x_chan,y_chan,dpi,nbins = 101,for_export=False,scatter_size=0.2):

    context = nullcontext() if for_export else plt.style.context("default")

    with context:
        if not for_export:
            plt.rcParams["font.family"] = "DejaVu Sans"
            
        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=fsize,dpi=dpi)

        frac_hist(f_sim_x,f_data_x,x_chan,ax1,nbins=nbins,for_export=for_export)
        frac_hist(f_sim_y,f_data_y,y_chan,ax2,nbins=nbins,for_export=for_export)
        frac_scatter(f_sim_x,f_sim_y,f_data_x,f_data_y,x_chan,y_chan,ax3,for_export,scatter_size=scatter_size)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, (ax1,ax2,ax3)

def get_empty(dpi, for_export=False):

    context = nullcontext() if for_export else plt.style.context("default")

    with context:
        if not for_export:
            plt.rcParams["font.family"] = "DejaVu Sans"

        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=fsize,dpi=dpi)
        for ax in [ax1,ax2,ax3]:
            ax.text(0.5,0.5,"No fit",transform=ax.transAxes)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig, (ax1,ax2,ax3)

def frac_hist(f_sim, f_data, chan_name, ax, nbins = 101, for_export=True):
    # Bounds for plotting
    low_lim = min([min(f_sim), min(f_data)])
    high_lim = max([max(f_sim), max(f_data)])
    range_lim = high_lim - low_lim
    low_lim -= 0.3*range_lim
    high_lim += 0.3*range_lim

    bins = np.linspace(low_lim, high_lim, nbins)
    binwidth = round(np.diff(bins)[0], 3)
    if binwidth == 0:
        binwidth = round(np.diff(bins)[0], 4)

    fsize = 35 if for_export else 13
    lblsize = 35 if for_export else 12

    ax.ticklabel_format(style='plain', axis='both')
    ax.tick_params(axis='both', labelsize=lblsize)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.hist(f_sim, bins, color = '#9999FF', label = 'Simulation', density = True)
    ax.hist(f_data, bins, color = 'black', histtype='step', label = 'Data', density = True)

    ax.set_xlabel(chan_name + 'OFamps / PSum', loc = 'right', fontsize=fsize)
    ax.set_ylabel('Probability / ' + str(binwidth), loc = 'top', fontsize=fsize)
    ax.grid(True, lw = 0.2, ls = '--', color = 'black')
    ax.legend(frameon=False,fontsize=fsize)

def frac_scatter(f_sim_x, f_sim_y, f_data_x, f_data_y, x_chan, y_chan, ax, for_export = True, scatter_size = 0.2):
    # Bounds for plotting
    low_lim = min([min(f_sim_x), min(f_sim_y), min(f_data_x), min(f_data_y)])
    high_lim = max([max(f_sim_x), max(f_sim_y), max(f_data_x), max(f_data_y)])
    range_lim = high_lim - low_lim
    low_lim -= 0.15*range_lim
    high_lim += 0.15*range_lim

    ax.scatter(f_sim_x, f_sim_y, s = scatter_size, marker = '.', color = '#9999FF', label= 'Simulation')
    ax.scatter(f_data_x, f_data_y, s = scatter_size, marker = '.', color = 'black', label = 'Data')

    fsize = 35 if for_export else 13
    lblsize = 35 if for_export else 12

    ax.set_xlabel(x_chan + 'OFamps / PSum', loc = 'right',fontsize=fsize)
    ax.set_ylabel(y_chan + 'OFamps / PSum', loc = 'top',fontsize=fsize)

    ax.set_xlim(low_lim, high_lim)
    ax.set_ylim(low_lim, high_lim)
    ax.grid(True, lw = 0.2, ls = '--', color = 'black')

    ax.ticklabel_format(style='plain', axis='both')
    ax.tick_params(axis='both', labelsize=lblsize)

    ax.legend(frameon=False,fontsize=fsize)

def frac_contour(f_sim_x, f_sim_y, f_data_x, f_data_y, x_chan, y_chan, dpi=100, savename=None):
    # Bounds for plotting
    low_lim = min([min(f_sim_x), min(f_sim_y), min(f_data_x), min(f_data_y)])
    high_lim = max([max(f_sim_x), max(f_sim_y), max(f_data_x), max(f_data_y)])
    range_lim = high_lim - low_lim
    low_lim -= 0.3*range_lim
    high_lim += 0.3*range_lim

    kde_sim = KernelDensity(kernel='gaussian', bandwidth=0.003).fit(np.vstack([f_sim_x, f_sim_y]).T)
    kde_data = KernelDensity(kernel='gaussian', bandwidth=0.003).fit(np.vstack([f_data_x, f_data_y]).T)

    x_grid = np.linspace(low_lim, high_lim, 100)
    y_grid = np.linspace(low_lim, high_lim, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)

    xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T

    z_sim_log = kde_sim.score_samples(xy_sample)
    z_data_log = kde_data.score_samples(xy_sample)
    z_sim = np.exp(z_sim_log).reshape(xx.shape)
    z_data = np.exp(z_data_log).reshape(xx.shape)

    plt.figure(figsize=(4,3), dpi=dpi)
    plt.contour(xx, yy, z_sim, levels = [1, 10, 30, 100, 300, 1000], colors = '#9999FF', linewidths = 1)
    plt.contour(xx, yy, z_data, levels = [1, 10, 30, 100, 300, 1000], colors = 'black', linewidths = 0.5)

    plt.xlabel(x_chan + 'OFamps / PSum', loc = 'right')
    plt.ylabel(y_chan + 'OFamps / PSum', loc = 'top')

    plt.xlim(low_lim, high_lim)
    plt.ylim(low_lim, high_lim)
    if savename is not None:
        if "." not in savename:
            savename += ".png"
            
        plt.savefig(savename, bbox_inches="tight")