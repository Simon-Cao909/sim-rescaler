from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Callable

import os
import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import copy
from sklearn.neighbors import KernelDensity
import ctypes
from array import array
from contextlib import nullcontext
import re
import ast
import random
import string

import ipywidgets as widgets
from IPython.display import display, clear_output

from tools.getters import get_ring_or_central, get_param_lt
from tools.validation import frac_plotter, frac_hist, frac_scatter, frac_contour, get_empty
from tools.transforms import quad_transform, legendre_quad_transform, power_transform, \
                       inverse_quad_transform, inverse_legendre_quad_transform

CHAN_LIST = ('PAS1','PBS1','PCS1','PDS1','PES1','PFS1',
             'PAS2','PBS2','PCS2','PDS2','PES2','PFS2')

def apply_cuts(df, cuts, cut_var, inclusive):
    '''
    Applies the given cuts to the dataframe
    Returns a dictionary of the cut dataframes
    '''
    if not cuts:
        return df

    cut_dfs = {}
    for label, cut in cuts.items():
        lo, hi = cut
        
        if inclusive:
            cut_arr = (df[cut_var] >= lo) & (df[cut_var] <= hi)
        else:
            cut_arr = (df[cut_var] > lo) & (df[cut_var] < hi)

        cut_dfs[label] = df[cut_arr]

    return cut_dfs

def nll_diff(params, x_arr, y_arr, oth_xy, fit, 
        setting_x, setting_y, 
        bandwidth, 
        rescale_fn,
        n_params_x,n_params_y):
    '''
    Calculates the negative log likelihood when applying the kde
    to the item being rescaled
    '''
    params = iter(params)

    if fit[0]:
        params_x = get_param_lt(params,n_params_x)
        x_arr = rescale_fn(x_arr,params_x,setting_x)
    
    if fit[1]:
        params_y = get_param_lt(params,n_params_y)
        y_arr = rescale_fn(y_arr,params_y,setting_y)
    
    xy_arr = np.column_stack((x_arr,y_arr))
    kde = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(xy_arr)
    logp = kde.score_samples(oth_xy)

    return -np.sum(logp)

def nll_same(params, kde, x_arr, y_arr, fit, 
             setting_x, setting_y,
             rescale_fn,
             n_params_x,n_params_y):
    '''
    Calculates the negative log likelihood when applying the kde
    to the target
    '''
    params = iter(params)
            
    if fit[0]:
        params_x = get_param_lt(params,n_params_x)
        x_arr = rescale_fn(x_arr, params_x, setting_x)
    if fit[1]:
        params_y = get_param_lt(params,n_params_y)
        y_arr = rescale_fn(y_arr, params_y, setting_y)
    
    xy_arr = np.column_stack((x_arr, y_arr))
    logp = kde.score_samples(xy_arr)
    
    return -np.sum(logp)

@dataclass
class SimRescaler:
    '''
    SimRescaler is a tool used to rescale DMC simulation to match real data

    It provides an interactive interface that can be used to guess parameters,
    fit parameters, and see validation plots

    Required Attributes
    -------------------
    - data (pd.DataFrame or dict): The data dataframe or a dictionary of data dataframes
    - sim (pd.DataFrame or dict): The simulation dataframe or a dictionary of simulation dataframes

    
    Semi-required attributes
    ------------------------
    - init_params (dict): A dictionary of the form {chan:[p1,p2,...],...}
                          This specifies the initial parameters used for rescaling
                          This is required for semiauto_fit(), 'Initial' in see_validation(), and export_params()
                          This is not required otherwise and can be determined through find_init_params()

    - n_params (dict): A dictionary of the form {chan:n_param,...}
                       This specifies the number of parameters for each channel
                       This is required for find_init_params() if self.init_params was not given

    - fitted_params (dict): A dictionary of the form {chan:[p1,p2,...],...}
                            This specifies the fitted parameters used for rescaling
                            This is required for 'Fitted' in see_validation() and for export_params()
                            This is not required otherwise and can be determined through semiauto_fit() or auto_fit()

    - errs (dict): A dictionary of the form {chan:{'parab':[err1,...],'+':[...],'-':[...],'gcc':[...]},...}
                   This specifies the errors for the fitting, including the parabolic error, +minos, -minos, and
                   global correlation coefficient
                   All but the gcc (that can be left as []) is required for export_params()


    Optional Attributes
    -------------------
    - channels (list or tuple, default=CHAN_LIST): The channels to be used

    - data_cuts (dict,default=None): A dictionary of the form {shell_name:(cut_lo,cut_hi),...}
                                     If data was given as a single dataframe, this will cut it
                                     into {shell_name:df_cut,...} and reassign it to self.data
    - data_cut_var (str,default=None): The variable used in the data cut (ex. PTOFamps)
    - data_cut_inclusive (bool,default=True): Whether to include the bounds in the cut

    - sim_cuts (dict,default=None): A dictionary of the form {shell_name:(cut_lo,cut_hi),...}
                                    If simulation was given as a single dataframe, this will cut it
                                    into {shell_name,df_cut} and reassign it to self.sim
    - sim_cut_var (str,default=None): The variable used in the simulation cut (ex. PTOFamps)
    - sim_cut_inclusive (bool,default=True): Whether to include the bounds in the cut

    - rescale_fn (callable,default=quad_transform): The rescaling function used
                                                    Must accept parameters (x,params,setting) in that order
                                                    - x is the variable to be rescaled
                                                    - params is the parameters used
                                                    - setting will be given by 'get_setting'. Default is 'ring' or 'central'
                                                        - This is used for treating different channels differently
    - inverse_fn (callable,default=inverse_quad_transform): The inverse of the rescaling function used
                                                            Must accept parameters (x,y,params,setting) in that order
                                                            - x is the parameter that was rescaled in the forward transform
                                                                - This is included only if you want to perform branch-cuts on
                                                                  non-monotonic functions
                                                            - y is the parameter to be rescaled in the inverse transform
                                                            - params is the parameters used
                                                            - setting will be given by get_setting. Default is 'ring' or 'central'
                                                                - This is used for treating different channels differently
                                                            * You can raise exceptions in this function for any reason
    - get_setting (callable,default=get_ring_or_central): A function used that should accept only the channel name (ex. PAS1)
                                                          and return a string describing the setting associated with that channel
    
    - fit_dict (dict,default=None): A dictionary of the form {(chan_1,chan_2):(0 or 1,0 or 1),...}
                                    This will be used in auto_fit only for the channels used for fitting
                                    and the order in which to fit it
                                    If you plan to use semiauto_fit, leave this as None
    
    - bandwidth (float,d    efault=0.003): The bandwidth used for the KDE when fitting
    - samples (int,default=1000): The number of samples used from the data and simulation to do the fitting

    - fit_to (str,default='data'): Must be either 'data' or 'sim'
                                   If 'data', the simulation will be rescaled
                                   If 'sim', the data will be rescaled
    - apply_kde_to (str,default='sim'): Must be either 'data' or 'sim'
                                        Controls which one to apply the KDE to
                                   
    - max_iter (int,default=1e5): The maximum number of calls for the minuit fit
    - tol (float,default=0.05): The tolerance for the minuit fit
    
    - nbins (int or dict,default=101): If int, this will be the number of bins used in the validation histograms
                                       If dict, it must be of the form {shell_name:num_bins,...}, where you can
                                       specify the number of bins for each shell
    - dpi (int,default=100): The dots per inch (resolution of the image) for the figures
    - xlim (dict or tuple,default=None): The xlim for the validation histograms
                                         This will also be the ylim of the scatter plot
                                         If dict, it should be of the form {shell_name1:xlim1,...}
                                         where each shell will have its own xlim
    - ylim (dict or tuple,default=None): The ylim for the validation histograms
                                         If dict, it should be of the form {shell_name1:ylim1,...}
                                         where each shell will have its own ylim

    - fitted_export_path (str,default="params.csv"): The export path of the fitted parameters (used in semiauto_fit)
    - init_export_path (str,default="init_params.csv"): The export path of the initial parameters (used in find_init_params)
    - img_dir (str,default=""): The directory to save the validation plots in (used in see_validation)
    '''
    data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]]
    simulation: Union[pd.DataFrame, Mapping[str, pd.DataFrame]]

    channels: Sequence[str] = CHAN_LIST
    
    data_cuts: Optional[Mapping[str, Tuple[float,float]]] = None
    data_cut_var: Optional[str] = None
    data_cut_inclusive: bool = True
    
    sim_cuts: Optional[Mapping[str, Tuple[float,float]]] = None
    sim_cut_var: Optional[str] = None
    sim_cut_inclusive: bool = True
    
    rescale_fn: Callable = quad_transform
    inverse_fn: Callable = inverse_quad_transform
    get_setting: Callable = get_ring_or_central

    init_params: Optional[Dict[str,Union[tuple,list]]] = None
    n_params: Optional[Dict[str,int]] = None
    fitted_params: Optional[Dict[str,Union[tuple,list]]] = None
    errs: Optional[Dict[str,Dict[str,Union[tuple,list]]]] = None
    bandwidth: float = 0.003
    samples: int = 1000
    fit_to: str = 'data'
    apply_kde_to: str = 'sim'
    bounds: Optional[Dict[str,list]] = None
    max_iter: int = 100000
    tol: float = 0.05
    fit_dict: Optional[Dict[Tuple[str,str],Tuple[int,int]]] = None

    nbins: Union[int,Dict[str,int]] = 101
    dpi: int = 100
    xlim: Optional[Union[Dict[str,Tuple[float,float]],Tuple[float,float]]] = None
    ylim: Optional[Union[Dict[str,Tuple[float,float]],Tuple[float,float]]] = None

    fitted_export_path: str = "params.csv"
    init_export_path: str = "init_params.csv"
    img_dir: str = ""

    def __post_init__(self):
        self.channels = tuple(c.upper() for c in self.channels)
        self.fit_to = self.fit_to.lower()

        if self.fit_to == 'simulation':
            self.fit_to = 'sim'
        elif self.fit_to == 'cute':
            self.fit_to = 'data'

        self.enabled: Dict[str,bool] = {c:False for c in self.channels}
        self.fitted: Dict[str, bool] = {c:False for c in self.channels}
        self.history: List[str] = []

        self.selected_chan2_1: str = self.channels[0]
        self.selected_chan2_2: str = self.channels[1]

        self.selected_chan3_1: str = self.channels[0]
        self.selected_chan3_2: str = self.channels[1]

        if isinstance(self.data, pd.DataFrame):
            if self.data_cut_var is None and self.data_cuts:
                raise ValueError("Data cut variable not given")
                
            self.data = apply_cuts(self.data,
                                   self.data_cuts,
                                   self.data_cut_var,
                                   self.data_cut_inclusive)

        if isinstance(self.simulation, pd.DataFrame):
            if self.sim_cut_var is None and self.sim_cuts:
                raise ValueError("Simulation cut variable not given")

            self.simulation = apply_cuts(self.simulation,
                                         self.sim_cuts,
                                         self.sim_cut_var,
                                         self.sim_cut_inclusive)
        
        if self.bounds is None:
            self.bounds = {}
            for chan in self.channels:
                if self.bounds.get(chan) is None:
                    self.bounds[chan] = []
                
                if self.init_params is not None:
                    params = self.init_params[chan]
                    for param in params:
                        bound = (0,3*param) if param > 0 else (3*param,0)
                else:
                    bound = (-10,10)
                
                self.bounds[chan].append(bound)
        
        if self.n_params is None and self.init_params is not None:
            self.n_params = {c:len(self.init_params[c]) for c in self.channels}

        if self.errs is None:
            self.errs = {c:{'parab':[],'+':[],'-':[],'gcc':[]} for c in self.channels}
        
        if self.fitted_params is None:
            self.fitted_params = {c:[] for c in self.channels}
        
        if self.fit_dict is None:
            self.fit_dict = {}
        
        if self.init_params is not None:
            for c in self.channels:
                if self.init_params.get(c) is None:
                    continue

                self.init_params[c] = list(self.init_params[c])
        
        self.fig: Optional[Figure] = None
        self.ax_hist_1: Optional[Axes] = None
        self.ax_hist_2: Optional[Axes] = None
        self.ax_hist_3: Optional[Axes] = None

        self.fig2: Optional[Figure] = None
        self.ax_hist2_1: Optional[Axes] = None
        self.ax_hist2_2: Optional[Axes] = None
        self.ax_hist2_3: Optional[Axes] = None

        self.fig3: Optional[Figure] = None
        self.ax_hist3_1: Optional[Axes] = None
        self.ax_hist3_2: Optional[Axes] = None
        self.ax_hist3_3: Optional[Axes] = None

        self.shell_names: list = list(self.data.keys()) if isinstance(self.data,dict) else ['Current']
        self.has_shells = False if len(self.shell_names) == 1 else True

        self.selected_shell: str = self.shell_names[0]
        self.selected_shell2: str = self.shell_names[0]
        self.selected_shell3: str = self.shell_names[0]

        self.x_sim: Optional[Union[np.ndarray,pd.Series]] = None
        self.x_data: Optional[Union[np.ndarray,pd.Series]] = None
        self.x_chan: Optional[str] = None
        self.x_setting: Optional[str] = None

        self.y_sim: Optional[Union[np.ndarray,pd.Series]] = None
        self.y_data: Optional[Union[np.ndarray,pd.Series]] = None
        self.y_chan: Optional[str] = None
        self.y_setting: Optional[str] = None

        self.fit: Optional[Tuple[int,int]] = None

        if ".csv" not in self.fitted_export_path:
            raise ValueError("fitted_export_path must have .csv")
        elif ".csv" not in self.init_export_path:
            raise ValueError("init_export_path must have .csv")

        self.img_extension: str = ".png"
        self.which_validation: str = "Initial"

        self.transform_setting: str = "Forward"

    def _get_shell_ind(self):
        return self.shell_names.index(self.selected_shell)

    def _get_frac(self,label,shell,x_chan,y_chan=None):
        if label == 'data':
            df = copy.deepcopy(self.data)
        elif label == 'sim':
            df = copy.deepcopy(self.simulation)
        
        if isinstance(df,dict):
            df = df[shell]

        return tuple(df[c+"OFamps"] / df['PSum'] \
                     for c in [x_chan, y_chan] if c is not None)

    def _rescale(self,x_sim_in,x_data_in,x_chan,which,
                 y_sim_in=None,y_data_in=None,y_chan=None):
        which = which.lower()

        x_sim, y_sim = copy.deepcopy(x_sim_in), copy.deepcopy(y_sim_in)
        x_data, y_data = copy.deepcopy(x_data_in), copy.deepcopy(y_data_in)

        x_setting = self.get_setting(x_chan)

        if y_chan is not None:
            y_setting = self.get_setting(y_chan)

        if which in ['fitted','fit']:
            x_params = self.fitted_params[x_chan]
            if y_chan is not None:
                y_params = self.fitted_params[y_chan]
        elif which in ['initial','init']:
            x_params = self.init_params[x_chan]
            if y_chan is not None:
                y_params = self.init_params[y_chan]

        if self.fit_to == 'data':
            x_sim = self.rescale_fn(x_sim,x_params,x_setting)
            if y_chan is not None:
                y_sim = self.rescale_fn(y_sim,y_params,y_setting)
        elif self.fit_to == 'sim':
            x_data = self.rescale_fn(x_data,x_params,x_setting)
            if y_chan is not None:
                y_data = self.rescale_fn(y_data,y_params,y_setting)
        
        if y_chan is not None:
            return x_sim, x_data, y_sim, y_data
        
        return x_sim, x_data

    def _inverse_rescale(self,x_sim_in,x_data_in,x_chan,which,
                         y_sim_in=None,y_data_in=None,y_chan=None):
        which = which.lower()
        x_sim, y_sim = copy.deepcopy(x_sim_in), copy.deepcopy(y_sim_in)
        x_data, y_data = copy.deepcopy(x_data_in), copy.deepcopy(y_data_in)

        x_setting = self.get_setting(x_chan)
        if y_chan is not None:
            y_setting = self.get_setting(y_chan)

        if which in ['fitted','fit']:
            x_params = self.fitted_params[x_chan]
            if y_chan is not None:
                y_params = self.fitted_params[y_chan]
        elif which in ['initial','init']:
            x_params = self.init_params[x_chan]
            if y_chan is not None:
                y_params = self.init_params[y_chan]

        if self.fit_to == 'data':
            x_data = self.inverse_fn(x_sim,x_data,x_params,x_setting)
            x_validate = self.inverse_fn(x_sim,
                                         self.rescale_fn(x_sim,x_params,x_setting),
                                         x_params,
                                         x_setting)
            x_validate_target = x_sim

            if y_chan is not None:
                y_data = self.inverse_fn(y_sim,y_data,y_params,y_setting)
                y_validate = self.inverse_fn(y_sim,
                                             self.rescale_fn(y_sim,y_params,y_setting),
                                             y_params,
                                             y_setting)
                y_validate_target = y_sim
        elif self.fit_to == 'sim':
            x_sim = self.inverse_fn(x_data,x_sim,x_params,x_setting)
            x_validate = self.inverse_fn(x_data,
                                         self.rescale_fn(x_data,x_params,x_setting),
                                         x_params,
                                         x_setting)
            x_validate_target = x_data

            if y_chan is not None:
                y_sim = self.inverse_fn(y_data,y_sim,y_params,y_setting)
                y_validate = self.inverse_fn(y_data,
                                             self.rescale_fn(y_data,y_params,y_setting),
                                             y_params,
                                             y_setting)
                y_validate_target = y_data

        if not np.allclose(x_validate,x_validate_target,atol=1e-9):
            raise Exception(f"Incorrect inverse for {x_chan}")

        if y_chan is not None:
            if not np.allclose(y_validate,y_validate_target,atol=1e-9):
                raise Exception(f"Incorrect inverse for {y_chan}")
            return x_sim, x_data, y_sim, y_data
        
        return x_sim, x_data

    def _get_nbins(self,shell):
        return self.nbins if isinstance(self.nbins,int) else self.nbins[shell]

    def _set_lims(self,ax1,ax2,ax3,shell):
        if self.xlim is not None:
            if isinstance(self.xlim,dict):
                xlim = self.xlim[shell]
            else:
                xlim = self.xlim

            for ax in [ax1,ax2,ax3]:
                ax.set_xlim(*xlim)

            ax3.set_ylim(*xlim)

        if self.ylim is not None:
            if isinstance(self.ylim,dict):
                ylim = self.ylim[shell]
            else:
                ylim = self.ylim

            for ax in [ax1,ax2]:
                ax.set_ylim(*ylim)
    

    ### INTERACTIVE 1 ###

    def semiauto_fit(self):
        '''
        Provides an interactive interface for parameter fitting
        Will use ROOT TMinuit for fitting and track uncertainties
        
        METHOD:
        1. Select two channels that you'd like to fit from the check boxes
        2. Press 'Fit'
        3. Once the fitting results have been printed, you can click on 'Validation plots'
           to see the validation plots
        4. Repeat
        5. Press 'Export' if you want to, which will create a .csv file containing
           the fitted parameters and uncertainties. The filepath was either specified
           during initialization as fitted_export_path or will be "params.csv"

        REQUIRES:
        - init_params
        - bounds
        '''
        self.output = widgets.Output()

        self.checkboxes = {
            chan: widgets.Checkbox(
                value = self.enabled.get(chan,False),
                description = chan,
                indent=False,
                layout = widgets.Layout(width="90px")
            )
         for chan in self.channels
        }

        for box in self.checkboxes.values():
            box.observe(self._on_box_change,names="value")
        
        self.shell_dropdown = widgets.Dropdown(
            options=list(self.shell_names),
            value=self.shell_names[self._get_shell_ind()],
            description="Shell:",
            disabled=not self.has_shells,
            layout = widgets.Layout(width="180px"),
        )
        self.shell_dropdown.observe(self._on_shell_change, names="value")

        apply_kde_to_dropdown = widgets.Dropdown(
            options=['data','sim'],
            value=self.apply_kde_to,
            description="Apply KDE to:",
            disabled=False,
            layout = widgets.Layout(width="180px")
        )
        apply_kde_to_dropdown.observe(self._on_apply_kde_to_change, names="value")

        fit_to_dropdown = widgets.Dropdown(
            options=['data','sim'],
            value=self.fit_to,
            description="Fit to:",
            disabled=False,
            layout=widgets.Layout(width="180px")
        )
        fit_to_dropdown.observe(self._on_fit_to_change,names="value")

        fit_button = widgets.Button(description="Fit")
        fit_button.on_click(lambda _: self._fit())

        clear_button = widgets.Button(description="Clear")
        clear_button.on_click(lambda _: self._clear())

        see_validation_button = widgets.Button(description="Validation plots")
        see_validation_button.on_click(lambda _: self._redraw_static())

        fit_info_button = widgets.Button(description="Fitting information")
        fit_info_button.on_click(lambda _: self._fitting_info())

        export_button = widgets.Button(description="Export")
        export_button.on_click(lambda _: self.export_params())

        self.channels_fitted_label = widgets.Label(value="Channels fitted:")

        controls = widgets.VBox([
            widgets.HBox([self.shell_dropdown,apply_kde_to_dropdown,fit_to_dropdown]),
            widgets.HTML("<b>Channels</b>"),
            widgets.HBox([self.checkboxes[c] for c in self.channels if c.endswith('1')]),
            widgets.HBox([self.checkboxes[c] for c in self.channels if c.endswith('2')]),
            widgets.HBox([fit_button,clear_button,see_validation_button,fit_info_button,export_button]),
            self.channels_fitted_label,
        ])

        display(widgets.VBox([controls,self.output]))

        self._redraw_static()

    def selected_chans(self) -> List[str]:
        return tuple(c for c in self.channels if self.enabled.get(c,False))

    def _on_box_change(self,change):
        owner = change["owner"]
        chan = None
        for ch, box in self.checkboxes.items():
            if box is owner:
                chan = ch
                break

        if chan is None:
            return
        
        old = bool(change["old"])
        new = bool(change["new"])
    
        if old == new:
            return

        self.enabled[chan] = new
        self.history.append(chan)

    def _on_shell_change(self,change):
        if not self.has_shells:
            return
        
        shell = change["new"]
        self.selected_shell = shell

        if self.x_chan is not None and self.y_chan is not None:
            self.x_sim, self.y_sim = self._get_frac('sim',self.selected_shell,self.x_chan,self.y_chan)
            self.x_data, self.y_data = self._get_frac('data',self.selected_shell,self.x_chan,self.y_chan)

        self._redraw_static()

    def _on_apply_kde_to_change(self,change):
        self.apply_kde_to = change['new']

    def _on_fit_to_change(self,change):
        self.fit_to = change['new']

    def _clear(self,redraw=True):
        self.fitted_params = {c:[] for c in self.channels}
        self.fitted = {c:False for c in self.channels}
        self.history = []
        self.errs = {c:{'parab':[],'+':[],'-':[],'gcc':[]} for c in self.channels}
        self.channels_fitted_label.value = "Channels fitted:"

        self.x_sim = None
        self.x_data = None
        self.x_chan = None
        self.x_setting = None

        self.y_sim = None
        self.y_data = None
        self.y_chan = None
        self.y_setting = None

        self.fig = None
        self.ax_hist_1 = None
        self.ax_hist_2 = None
        self.ax_hist_3 = None

        if redraw:
            self._redraw_static()
        else:
            self.enabled = {c:False for c in self.channels}
    
    def _fitting_info(self):
        with self.output:
            print("=" * 80)
            print("SIMRESCALER FIT REPORT")
            print("=" * 80)

            print(f"Fit target : {self.fit_to}")
            print(f"Shell      : {self.selected_shell}")
            print()

            print("FIT PAIRS")
            print("-" * 80)

            if len(self.fit_dict) == 0:
                print("No fits performed.")
            else:
                for i, ((x_chan, y_chan), fit) in enumerate(self.fit_dict.items(), start=1):
                    fit_labels = []

                    if fit[0]:
                        fit_labels.append(x_chan)
                    if fit[1]:
                        fit_labels.append(y_chan)

                    fit_labels = ", ".join(fit_labels)

                    print(
                        f"{i:2d}. ({x_chan}, {y_chan}) "
                        f"-> floated [{fit_labels}]"
                    )

            print()
            print("=" * 80)
            print("CHANNEL SUMMARY")
            print("=" * 80)

            for chan in self.channels:

                setting = self.get_setting(chan)

                init = None
                if self.init_params is not None:
                    init = self.init_params.get(chan)

                bounds = None
                if self.bounds is not None:
                    bounds = self.bounds.get(chan)

                fitted = self.fitted_params.get(chan, [])
                fitted_flag = self.fitted.get(chan, False)

                npars = len(init) if init is not None else 0

                print()
                print(f"{chan}")
                print("-" * 40)

                print(f"Setting     : {setting}")
                print(f"Fit status  : {'FIT' if fitted_flag else 'NOT FIT'}")

                print("\nInitial parameters")
                if init is None:
                    print("  None")
                else:
                    for i, val in enumerate(init):
                        print(f"  p{i}: {val}")

                print("\nBounds")
                if bounds is None:
                    print("  None")
                else:
                    for i, (lo, hi) in enumerate(bounds):
                        print(f"  p{i}: [{lo}, {hi}]")

                print("\nFitted parameters")

                if not fitted_flag:
                    for i in range(npars):
                        print(f"  p{i}: nan")
                else:
                    parab = self.errs[chan]["parab"]
                    err_p = self.errs[chan]["+"]
                    err_m = self.errs[chan]["-"]
                    gcc = self.errs[chan]["gcc"]

                    for i, val in enumerate(fitted):
                        print(f"  p{i}: {val}")

                    print("\nUncertainties")
                    print(f"  Parabolic : {parab}")
                    print(f"  +MINOS    : {err_p}")
                    print(f"  -MINOS    : {err_m}")
                    print(f"  GCC       : {gcc}")

            print()
            print("=" * 80)

    def export_params(self):
        '''
        Exports the initial parameters, fitted parameters, and uncertainties

        REQUIRES:
        - init_params (updated through find_init_params())
        - fitted_params (updated through semiauto_fit() or auto_fit())
        - errs (updated through semiauto_fit() or auto_fit())
        - bounds
        '''
        df = pd.DataFrame(index=self.channels)
        most_params = max(len(p) for p in self.init_params.values())

        def safe_get(lt,ind,fallback="",prefix="",suffix=""):
            if ind < len(lt):
                return prefix+f"{lt[ind]:.4f}"+suffix
            else:
                return fallback

        chans_used_labels = []
        for chan in self.channels:
            found = False
            for (fit_set1,fit_set2), (which_fit1, which_fit2) in self.fit_dict.items():
                if chan == fit_set1 and which_fit1:
                    oth_chan = fit_set2
                    extra = "flt" if which_fit2 else "fix"
                    found = True
                    break
                elif chan == fit_set2 and which_fit2:
                    oth_chan = fit_set1
                    extra = "flt" if which_fit1 else "fix"
                    found = True
                    break
                
            if found:
                chan_let = chan[1]
                oth_chan_let = oth_chan[1]
                if chan_let < oth_chan_let:
                    chans_used_label = f"{chan} (flt), {oth_chan} ({extra})"
                else:
                    chans_used_label = f"{oth_chan} ({extra}), {chan} (flt)"
            else:
                chans_used_label = "Not fitted"
            
            chans_used_labels.append(chans_used_label)
        
        df["Channels used"] = chans_used_labels

        for i in range(most_params):
            p0s = [safe_get(params,i) for params in self.init_params.values()]
            ps = [safe_get(params,i) for params in self.fitted_params.values()]
            parab_uncert = [safe_get(uncert['parab'],i,prefix=" ±") for uncert in self.errs.values()]
            minus_uncert = [safe_get(uncert['-'],i,prefix=" ",suffix="/") for uncert in self.errs.values()]
            plus_uncert = [safe_get(uncert['+'],i,prefix="+") for uncert in self.errs.values()]
            for ind, p in enumerate(ps):
                ps[ind] = p + parab_uncert[ind] + minus_uncert[ind] + plus_uncert[ind]

            df[f"p{i}"] = ps
            df[f"p{i}_0"] = p0s
            df[f"p{i}_bounds"] = [
                f"[{b[i][0]}, {b[i][1]}]"
                if i < len(b)
                else ""
                for b in self.bounds.values()
            ]
        
        df.to_csv(self.fitted_export_path)

    def _fit(self,verbose=2,update_fit_dict=True):
        with self.output:
            clear_output(wait=True)
            
            if len(self.selected_chans()) != 2:
                print("Only select two channels!")
                return
            if self.init_params is None:
                print("No initial parameters given!")
                return
    
            x_chan, y_chan = self.selected_chans()
    
            x0, y0 = list(self.init_params.get(x_chan)),list(self.init_params.get(y_chan))
            if x0 is None:
                print(f"No initial parameters given for {x_chan}!")
                return
            if y0 is None:
                print(f"No initial parameters given for {y_chan}!")
                return
    
            x_sim, y_sim = self._get_frac('sim',self.selected_shell,x_chan,y_chan)
            x_data, y_data = self._get_frac('data',self.selected_shell,x_chan,y_chan)
    
            x_setting, y_setting = self.get_setting(x_chan), self.get_setting(y_chan)
            x_bounds, y_bounds = self.bounds.get(x_chan), self.bounds.get(y_chan)
    
            self.x_sim = copy.deepcopy(x_sim)
            self.x_data = copy.deepcopy(x_data)
            self.x_chan = x_chan
            self.x_setting = x_setting
    
            self.y_sim = copy.deepcopy(y_sim)
            self.y_data = copy.deepcopy(y_data)
            self.y_chan = y_chan
            self.y_setting = y_setting
    
            if x_bounds is None:
                print(f"No bounds given for {x_chan}!")
                return
            if y_bounds is None:
                print(f"No bounds given for {y_chan}!")
                return
    
            if self.fitted.get(x_chan,False) and self.fitted.get(y_chan,False):
                print("No fitting done! Both channels have already been fit")
                return
            
            if verbose >= 1:
                if self.fitted.get(x_chan):
                    print(f"Fitting {y_chan} with {x_chan} fixed...")
                elif self.fitted.get(y_chan):
                    print(f"Fitting {x_chan} with {y_chan} fixed...")
                else:
                    print(f"Fitting {x_chan} and {y_chan} together...")
            
            
            if self.apply_kde_to == self.fit_to:
                init_params, bounds, fit, fcn = self._fit_same(x_data,y_data,
                                                               x_sim,y_sim,
                                                               x_bounds,y_bounds,
                                                               x0,y0,
                                                               x_chan,y_chan,
                                                               x_setting,y_setting,
                                                               verbose=verbose)
            else:
                init_params, bounds, fit, fcn = self._fit_diff(x_data,y_data,
                                                               x_sim,y_sim,
                                                               x_bounds,y_bounds,
                                                               x0,y0,
                                                               x_chan,y_chan,
                                                               x_setting,y_setting,
                                                               verbose=verbose)
            
            m = ROOT.TMinuit(len(init_params))
            m.SetFCN(fcn)

            ierflg = ctypes.c_int(0)

            for i in range(len(init_params)):
                lo_bound = bounds[i][0]
                hi_bound = bounds[i][1]

                step_size = (hi_bound - lo_bound)/1000
                m.mnparm(i,f"{i}",init_params[i],step_size,
                            lo_bound,hi_bound,ierflg)

            arglist = array('d',[0.5])
            m.mnexcm("SET ERR", arglist, 1, ierflg)

            arglist = array('d',[self.max_iter,self.tol])
            m.mnexcm("MIGRAD", arglist, 2, ierflg)
            m.mnexcm("MINOS", arglist, 2, ierflg)

            for i in range(len(init_params)):
                val = ctypes.c_double(0)
                err = ctypes.c_double(0)
                err_plus = ctypes.c_double(0)
                err_minus = ctypes.c_double(0)
                err_parab = ctypes.c_double(0)
                gcc = ctypes.c_double(0)

                m.GetParameter(i,val,err)
                m.mnerrs(i,err_plus,err_minus,err_parab,gcc)

                if fit[0] and fit[1]:
                    if i < len(x0):
                        edit_c = x_chan
                    else:
                        edit_c = y_chan
                elif fit[0]:
                    edit_c = x_chan
                elif fit[1]:
                    edit_c = y_chan
                
                self.fitted_params[edit_c].append(val.value)
                
                found_errs = [err_parab,err_plus,err_minus,gcc]
                for err_label,err in zip(self.errs[edit_c].keys(),found_errs):
                    self.errs[edit_c][err_label].append(err.value)
            
            amin = ctypes.c_double(0)
            edm = ctypes.c_double(0)
            errdef = ctypes.c_double(0)
            nvpar = ctypes.c_int(0)
            nparx = ctypes.c_int(0)
            icstat = ctypes.c_int(0)

            m.mnstat(amin, edm, errdef, nvpar, nparx, icstat)

            npar = len(init_params)
            cov_arr = array('d',[0.]*(npar*npar))
            m.mnemat(cov_arr,npar)
            cov = np.array(cov_arr).reshape((npar,npar))
            sigma = np.sqrt(np.diag(cov))
            cor = cov/np.outer(sigma,sigma)

            if verbose >= 1:
                print()
                print("FITTING RESULTS")
                print("===============")
                if verbose >= 2:
                    print("FCN =", amin.value)
                    print("EDM =", edm.value)
                    print("Cov status =", icstat.value)
                    print()
                    print("Covariance matrix")
                    print("-----------------")
                    print(cov)
                    print()
                    print("Correlation matrix")
                    print("------------------")
                    print(cor)
                    print()
                print(f"Best params found for {x_chan}:", self.fitted_params[x_chan])
                print(f"Uncertainties for {x_chan}:")
                print("- Parabolic:",self.errs[x_chan]['parab'])
                print("- +MINOS",self.errs[x_chan]['+'])
                print("- -MINOS",self.errs[x_chan]['-'])
                print("- GCC",self.errs[x_chan]['gcc'])
                print()
                print(f"Best params found for {y_chan}:", self.fitted_params[y_chan])
                print(f"Uncertainties for {y_chan}:")
                print("- Parabolic:",self.errs[y_chan]['parab'])
                print("- +MINOS",self.errs[y_chan]['+'])
                print("- -MINOS",self.errs[y_chan]['-'])
                print("- GCC",self.errs[y_chan]['gcc'])

            self.fit = fit
            if update_fit_dict:
                self.fit_dict[(x_chan,y_chan)] = fit

            if not self.fitted.get(x_chan,False):
                self.channels_fitted_label.value += f" {x_chan},"
            if not self.fitted.get(y_chan,False):
                self.channels_fitted_label.value += f" {y_chan},"

            self.fitted[x_chan] = True
            self.fitted[y_chan] = True

    def _fit_same(self,
                  x_data,y_data,
                  x_sim,y_sim,
                  x_bounds,y_bounds,
                  x0,y0,
                  x_chan,y_chan,
                  x_setting,y_setting,
                  verbose=2):
        bounds = []
        samples = self.samples
        init_params = []

        if self.fit_to == 'data':
            xy_data = np.column_stack((x_data[:samples],y_data[:samples]))
            kde = KernelDensity(kernel="gaussian",bandwidth=self.bandwidth).fit(xy_data)
        elif self.fit_to == 'sim':
            xy_sim = np.column_stack((x_sim[:samples],y_sim[:samples]))
            kde = KernelDensity(kernel="gaussian",bandwidth=self.bandwidth).fit(xy_sim)
        
        fit = [1,1]
        if self.fitted.get(x_chan,False):
            x_sim, x_data = self._rescale(x_sim,x_data,x_chan,"fit")
            fit[0] = 0
        else:
            init_params.extend(x0)
            bounds += list(x_bounds)
        
        if self.fitted.get(y_chan,False):
            y_sim, y_data = self._rescale(y_sim,y_data,y_chan,"fit")
            fit[1] = 0
        else:
            init_params.extend(y0)
            bounds += list(y_bounds)
        
        if self.fit_to == 'data':
            x_arr, y_arr = x_sim, y_sim
        elif self.fit_to == 'sim':
            x_arr, y_arr = x_data, y_data
        
        if verbose >= 1:
            print("Initial params:",init_params)
            print("Bounds:", bounds)
            print("Initial NLL",nll_same(init_params,
                                         kde,
                                         x_arr[:samples],
                                         y_arr[:samples],
                                         fit,
                                         x_setting,
                                         y_setting,
                                         self.rescale_fn,
                                         len(x0),len(y0)))
        
        def fcn(npar, gin, f, par, iflag):
            params = [par[i] for i in range(len(init_params))]

            val = nll_same(
                params,
                kde,
                x_arr[:samples],
                y_arr[:samples],
                fit,
                x_setting,
                y_setting,
                self.rescale_fn,
                len(x0),len(y0)
            )

            f.value = val
        
        return init_params, bounds, fit, fcn

    def _fit_diff(self,
                  x_data,y_data,
                  x_sim,y_sim,
                  x_bounds,y_bounds,
                  x0,y0,
                  x_chan,y_chan,
                  x_setting,y_setting,
                  verbose=2):
        bounds = []
        samples = self.samples
        init_params = []

        fit = [1,1]
        if self.fitted.get(x_chan,False):
            x_sim,x_data = self._rescale(x_sim,x_data,x_chan,which="fit")
            fit[0] = 0
        else:
            init_params.extend(x0)
            bounds += list(x_bounds)

        if self.fitted.get(y_chan,False):
            y_sim,y_data = self._rescale(y_sim,y_data,y_chan,which="fit")
            fit[1] = 0
        else:
            init_params.extend(y0)
            bounds += list(y_bounds)
        
        if self.fit_to == 'data':
            x_arr, y_arr = x_sim[:samples], y_sim[:samples]
            oth_xy = np.column_stack((x_data[:samples],y_data[:samples]))
        elif self.fit_to == 'sim':
            x_arr, y_arr = x_data[:samples], y_data[:samples]
            oth_xy = np.column_stack((x_sim[:samples],y_sim[:samples]))

        if verbose >= 1:
            print("Initial params:",init_params)
            print("Bounds:", bounds)
            print("Initial NLL",nll_diff(init_params,
                                         x_arr,
                                         y_arr,
                                         oth_xy,
                                         fit,
                                         x_setting,
                                         y_setting,
                                         self.bandwidth,
                                         self.rescale_fn,
                                         len(x0),len(y0)))

        def fcn(npar,gin,f,par,iflag):
            params = [par[i] for i in range(len(init_params))]

            val = nll_diff(params,
                        x_arr,
                        y_arr,
                        oth_xy,
                        fit,
                        x_setting,
                        y_setting,
                        self.bandwidth,
                        self.rescale_fn,
                        len(x0),len(y0))
            
            f.value = val
        
        return init_params, bounds, fit, fcn

    def _redraw_static(self):
        with self.output:
            clear_output(wait=True)

            if any(el is None for el in [self.x_sim,self.x_data,
                                         self.y_sim,self.y_data,
                                         self.x_chan,self.y_chan,
                                         self.x_setting,self.y_setting,
                                         self.fit]):
                self.fig, (self.ax_hist_1,
                           self.ax_hist_2,
                           self.ax_hist_3) = get_empty(self.dpi)
            else:
                x_sim,x_data,y_sim,y_data = self._rescale(self.x_sim,self.x_data,self.x_chan,"fit",
                                                          self.y_sim,self.y_data,self.y_chan)

                num_bins = self._get_nbins(self.selected_shell)
                self.fig, (self.ax_hist_1,
                           self.ax_hist_2,
                           self.ax_hist_3) = frac_plotter(x_sim,y_sim,
                                                         x_data,y_data,
                                                         self.x_chan,self.y_chan,
                                                         self.dpi,num_bins)
            
            self._set_lims(self.ax_hist_1,
                           self.ax_hist_2,
                           self.ax_hist_3,
                           self.selected_shell)

            self.fig.suptitle("Validation plots",fontsize=14)
            plt.show()
    

    ### INTERACTIVE 2 ###

    def find_init_params(self):
        '''
        Provides an interactive interface for finding the initial parameters
        
        METHOD:
        1. Select the channels you'd like to fit from the dropdown
        2. Edit the numbers in the boxes and see the graph change
           The initial parameters will automatically be saved
        3. Choose another channel pair and repeat
        4. Press the 'export' button when you are done
           The filepath is init_export_path specified on initialization
           If none was given, it is init_params.csv
        
        REQUIRES:
        - n_params if init_params was not given
        '''
        self.output2 = widgets.Output()

        if self.n_params is None:
            raise ValueError("n_params cannot be None!")
        
        self.chan_dropdown2_1 = widgets.Dropdown(
            options=list(self.channels),
            value=self.selected_chan2_1,
            description="Channel 1:",
            disabled=False,
            layout=widgets.Layout(width="180px"),
        )
        self.chan_dropdown2_1.observe(lambda change: self._on_chan_change2(change,1),names="value")

        self.chan_dropdown2_2 = widgets.Dropdown(
            options=list(self.channels),
            value=self.selected_chan2_2,
            description="Channel 2:",
            disabled=False,
            layout=widgets.Layout(width="180px"),
        )
        self.chan_dropdown2_2.observe(lambda change: self._on_chan_change2(change,2),names="value")

        self.shell_dropdown2 = widgets.Dropdown(
            options=list(self.shell_names),
            value=self.selected_shell2,
            description="Shell:",
            disabled=not self.has_shells,
            layout = widgets.Layout(width="180px"),
        )
        self.shell_dropdown2.observe(self._on_shell_change2, names="value")

        max_n_params = max(self.n_params.values())
        if self.init_params is None:
            self.init_params = {c:[1.0 for _ in range(self.n_params[c])] for c in self.channels}
        
        for c in self.channels:
            if self.init_params.get(c) is None:
                self.init_params[c] = [1.0 for _ in range(self.n_params[c])]

        def safe_get(lt,ind):
            if ind < len(lt):
                return lt[ind]
            else:
                return 0.0
        
        self.param_inputs_1 = [widgets.FloatText(value=safe_get(self.init_params[self.selected_chan2_1],i),
                                                 description=(f"CH1 p{i}:" if i == 0 else f"p{i}"),
                                                 disabled=False,
                                                 continuous_update=False) 
                             for i in range(max_n_params)]
        
        self.param_inputs_2 = [widgets.FloatText(value=safe_get(self.init_params[self.selected_chan2_2],i),
                                                 description=(f"CH2 p{i}:" if i == 0 else f"p{i}"),
                                                 disabled=False,
                                                 continuous_update=False) 
                             for i in range(max_n_params)]
        
        for i1, p_input1 in enumerate(self.param_inputs_1):
            p_input1.observe(lambda change, i=i1:self._on_param_input_change(change,i,1),names="value")
        
        for i2, p_input2 in enumerate(self.param_inputs_2):
            p_input2.observe(lambda change, i=i2:self._on_param_input_change(change,i,2),names="value")

        update_like_button = widgets.Button(description="Use same for like channels")
        update_like_button.on_click(lambda _: self._use_same_for_like_channels())

        update_oth_side_button = widgets.Button(description="Use same for other side")
        update_oth_side_button.on_click(lambda _: self._use_same_for_other_side())

        guess_bounds_button = widgets.Button(description="Guess bounds")
        guess_bounds_button.on_click(lambda _: self._guess_bounds())

        export_button = widgets.Button(description="Export")
        export_button.on_click(lambda _: self._export_init_params())

        self.controls2 = widgets.VBox([
            self.shell_dropdown2,
            widgets.HBox([self.chan_dropdown2_1,self.chan_dropdown2_2]),
            widgets.HBox(self.param_inputs_1),
            widgets.HBox(self.param_inputs_2),
            widgets.HBox([update_like_button,update_oth_side_button,guess_bounds_button,export_button]),
        ])

        display(widgets.VBox([self.controls2,self.output2]))

        self._redraw_static2()

    def _on_chan_change2(self,change,i):
        chan = change['new']

        def safe_get(lt,ind):
            if ind < len(lt):
                return lt[ind]
            else:
                return 0.0

        if i == 1:
            self.selected_chan2_1 = chan
            for ind,p in enumerate(self.param_inputs_1):
                p.value = safe_get(self.init_params[chan],ind)
        elif i == 2:
            self.selected_chan2_2 = chan
            for ind, p in enumerate(self.param_inputs_2):
                p.value = safe_get(self.init_params[chan],ind)

        
        self._redraw_static2()

    def _on_param_input_change(self,change,ind,ind2):
        new_val = change['new']
        old_val = change['old']
        
        if new_val == old_val:
            return

        if ind2 == 1:
            if ind >= len(self.init_params[self.selected_chan2_1]):
                return
            self.init_params[self.selected_chan2_1][ind] = new_val
            
        elif ind2 == 2:
            if ind >= len(self.init_params[self.selected_chan2_2]):
                return
            self.init_params[self.selected_chan2_2][ind] = new_val
        
        self._redraw_static2()

    def _on_shell_change2(self,change):
        if not self.has_shells:
            return
        
        shell = change['new']
        self.selected_shell2 = shell

        self._redraw_static2()

    def _use_same_for_like_channels(self):
        current_params_1 = self.init_params[self.selected_chan2_1]
        current_setting_1 = self.get_setting(self.selected_chan2_1)

        current_params_2 = self.init_params[self.selected_chan2_2]
        current_setting_2 = self.get_setting(self.selected_chan2_2)

        for c in self.channels:
            if c == self.selected_chan2_1 or c == self.selected_chan2_2:
                continue

            setting = self.get_setting(c)

            if setting == current_setting_1:
                self.init_params[c] = current_params_1
            elif setting == current_setting_2:
                self.init_params[c] = current_params_2

    def _use_same_for_other_side(self):
        current_params_1 = self.init_params[self.selected_chan2_1]
        current_params_2 = self.init_params[self.selected_chan2_2]

        side_1 = int(self.selected_chan2_1[-1])
        letters_1 = self.selected_chan2_1[:-1]

        side_2 = int(self.selected_chan2_2[-1])
        letters_2 = self.selected_chan2_2[:-1]

        oth_side_1 = 1 if side_1 == 2 else 2
        oth_side_2 = 1 if side_2 == 1 else 2

        oth_chan_1 = f"{letters_1}{oth_side_1}"
        oth_chan_2 = f"{letters_2}{oth_side_2}"

        self.init_params[oth_chan_1] = current_params_1
        self.init_params[oth_chan_2] = current_params_2

    def _guess_bounds(self):
        self.bounds = {}
        for c in self.channels:
            self.bounds[c] = []
            params = self.init_params[c]
            for param in params:

                if param > 0:
                    bound = (0,3*param)
                elif param < 0:
                    bound = (3*param,0)
                else:
                    bound = (-5,5)

                self.bounds[c].append(bound)
    
    def _export_init_params(self):
        df = pd.DataFrame(index=self.channels)
        most_params = max(len(p) for p in self.init_params.values())

        def safe_get(lt,ind,fallback="",prefix="",suffix=""):
            if ind < len(lt):
                return prefix+f"{lt[ind]:.4f}"+suffix
            else:
                return fallback
        
        for i in range(most_params):
            p0s = [safe_get(params,i) for params in self.init_params.values()]
            df[f"p{i}_0"] = p0s

            df[f"p{i}_bounds"] = [
                f"[{b[i][0]}, {b[i][1]}]"
                if i < len(b)
                else ""
                for b in self.bounds.values()
            ]
        
        df.to_csv(self.init_export_path)

    def _redraw_static2(self):
        with self.output2:
            clear_output(wait=True)

            sim_1, sim_2 = self._get_frac("sim",self.selected_shell2,self.selected_chan2_1,self.selected_chan2_2)
            data_1, data_2 = self._get_frac("data",self.selected_shell2,self.selected_chan2_1,self.selected_chan2_2)
            
            sim_1, data_1, sim_2, data_2 = self._rescale(sim_1,data_1,self.selected_chan2_1,"init",
                                                         sim_2,data_2,self.selected_chan2_2)
            
            num_bins = self._get_nbins(self.selected_shell2)
            
            self.fig2, (self.ax_hist2_1,
                        self.ax_hist2_2,
                        self.ax_hist2_3) = frac_plotter(sim_1,sim_2,
                                                        data_1,data_2,
                                                        self.selected_chan2_1,self.selected_chan2_2,
                                                        self.dpi,num_bins)

            self._set_lims(self.ax_hist2_1,
                           self.ax_hist2_2,
                           self.ax_hist2_3,
                           self.selected_shell2)

            self.fig2.suptitle("Validation plots",fontsize=14)
            plt.show()
        
    
    ### INTERACTIVE 3 ###

    def see_validation(self,add_random=False):
        '''
        Provides an interactive interface for viewing validation plots

        :param add_random (bool): If True, adds a random string of letters after the exported file name
                                  This is so, when adding to confluence, the images do not 
                                  overwrite each other
                                  If False, nothing is done

        METHOD:
        1. Select the type of validation plot you want to see from the dropdown
            - Fitted: For fitted parameters scaling
            - Initial: For initial parameters scaling
            - No rescale: For no rescaling
        2. Select the channels from the dropdown that you'd like to see the validation plots of
        3. Click 'Export current' to export the images of the current plots
           or 'Export all' to export the images for all channels
        
        REQUIRES:
        - fitted_params if you want to see the plots with fitted rescaling
        - init_params if you want to see the plots with initial rescaling
        - fit_dict (created with semiauto_fit() or in initialization) if you want to export all scatter plots
        '''
        self.add_random = add_random

        self.output3 = widgets.Output()

        self.chan_dropdown3_1 = widgets.Dropdown(
            options=list(self.channels),
            value=self.selected_chan3_1,
            description="Channel 1:",
            disabled=False,
            layout=widgets.Layout(width="180px"),
        )
        self.chan_dropdown3_1.observe(lambda change: self._on_chan_change3(change,1),names="value")

        self.chan_dropdown3_2 = widgets.Dropdown(
            options=list(self.channels),
            value=self.selected_chan3_2,
            description="Channel 2:",
            disabled=False,
            layout=widgets.Layout(width="180px"),
        )
        self.chan_dropdown3_2.observe(lambda change: self._on_chan_change3(change,2),names="value")

        self.shell_dropdown3 = widgets.Dropdown(
            options=list(self.shell_names),
            value=self.selected_shell3,
            description="Shell:",
            disabled=not self.has_shells,
            layout = widgets.Layout(width="180px"),
        )
        self.shell_dropdown3.observe(self._on_shell_change3, names="value")

        self.img_extension_dropdown = widgets.Dropdown(
            options=[".png",".pdf",".svg",".jpg"],
            value=self.img_extension,
            description="Extension:",
            disabled=False,
            layout = widgets.Layout(width="180px")
        )
        self.img_extension_dropdown.observe(self._on_img_extension_change,names="value")

        self.which_validation_dropdown = widgets.Dropdown(
            options=['Fitted','Initial','No rescale'],
            value=self.which_validation,
            description="Type:",
            disabled=False,
            layout=widgets.Layout(width="180px")
        )
        self.which_validation_dropdown.observe(self._on_validation_dropdown_change,names="value")

        transform_setting_dropdown = widgets.Dropdown(
            options=['Forward','Inverse'],
            value=self.transform_setting,
            description="Transform:",
            disabled=False,
            layout=widgets.Layout(width="180px")
        )
        transform_setting_dropdown.observe(self._on_transform_setting_change,names="value")

        export_curr_button = widgets.Button(description="Export current")
        export_all_button = widgets.Button(description="Export all")

        export_curr_button.on_click(lambda _: self._export_validation("current"))
        export_all_button.on_click(lambda _: self._export_validation("all"))

        controls = widgets.VBox([
            widgets.HBox([self.shell_dropdown3,self.which_validation_dropdown,transform_setting_dropdown]),
            widgets.HBox([self.chan_dropdown3_1,self.chan_dropdown3_2]),
            widgets.HBox([export_curr_button,export_all_button,self.img_extension_dropdown])
        ])

        display(widgets.VBox([controls,self.output3]))

        self._redraw_static3()
    
    def _on_chan_change3(self,change,i):
        chan = change['new']

        if i == 1:
            self.selected_chan3_1 = chan
        elif i == 2:
            self.selected_chan3_2 = chan
        
        self._redraw_static3()
    
    def _on_shell_change3(self,change):
        if not self.has_shells:
            return
        
        shell = change['new']
        self.selected_shell3 = shell

        self._redraw_static3()
    
    def _on_img_extension_change(self,change):
        extension = change['new']
        self.img_extension = extension

    def _on_validation_dropdown_change(self,change):
        self.which_validation = change['new']

        self._redraw_static3()

    def _on_transform_setting_change(self,change):
        self.transform_setting = change['new']
        self._redraw_static3()

    def _export_validation(self,which):
        rescaler = self._rescale if self.transform_setting == 'Forward' else self._inverse_rescale

        if which == 'current':
            chans_to_export = [self.selected_chan3_1,self.selected_chan3_2]
            scatter_pairs = [(self.selected_chan3_1,self.selected_chan3_2)]
        elif which == 'all':
            chans_to_export = self.channels
            scatter_pairs = list(self.fit_dict.keys()) if self.fit_dict else None
        
        shell = self.selected_shell3

        rescale = False if self.which_validation == 'No rescale' else True
        for chan in chans_to_export:
            sim, = self._get_frac('sim',shell,chan)
            data, = self._get_frac('data',shell,chan)

            params_exist = True
            
            if self.which_validation == 'Fitted':
                params = self.fitted_params[chan]
                if params == []:
                    params_exist = False
            elif self.which_validation == 'Initial':
                if self.init_params is None:
                    params_exist = False
                else:
                    params = self.init_params.get(chan)
                    if params is None:
                        params_exist = False
            
            if rescale and params_exist:
                try:
                    sim, data = rescaler(sim,data,chan,self.which_validation)
                except Exception as e:
                    print("Encountered exception when trying to export validation plots for "
                          f"{chan}: {e}. Skipping...")
                    continue
            
            if params_exist:
                num_bins = self._get_nbins(self.selected_shell3)
                fig,ax = plt.subplots()

                frac_hist(sim,data,chan,ax,num_bins)

                if self.xlim is not None:
                    xlim = self.xlim if isinstance(self.xlim,tuple) else self.xlim[self.selected_shell3]
                    ax.set_xlim(*xlim)
                if self.ylim is not None:
                    ylim = self.ylim if isinstance(self.ylim,tuple) else self.ylim[self.selected_shell3]
                    ax.set_ylim(*ylim)

                if self.add_random:
                    s = ''.join(random.choices(string.ascii_lowercase, k=6))
                else:
                    s = ''
                fig.savefig(os.path.join(self.img_dir,f"{chan}_hist{s}{self.img_extension}"))
        
        if scatter_pairs is None:
            return

        for (chan_1,chan_2) in scatter_pairs:
            sim_1, sim_2 = self._get_frac('sim',shell,chan_1,chan_2)
            data_1, data_2 = self._get_frac('data',shell,chan_1,chan_2)

            if self.which_validation == 'Fitted':
                params_1 = self.fitted_params[chan_1]
                params_2 = self.fitted_params[chan_2]
                if params_1 == [] or params_2 == []:
                    params_exist = False
            elif self.which_validation == 'Initial':
                if self.init_params is None:
                    params_exist = False
                else:
                    params_1 = self.init_params.get(chan_1)
                    params_2 = self.init_params.get(chan_2)
                    if params_1 is None or params_2 is None:
                        params_exist = False
            elif self.which_validation == 'No rescale':
                rescale = False
            
            if rescale and params_exist:
                try:
                    sim_1, data_1, sim_2, data_2 = rescaler(sim_1,data_1,chan_1,self.which_validation,
                                                            sim_2,data_2,chan_2)
                except Exception as e:
                    print("Encountered exception when trying to export scatter plot for "
                          f"{chan_1} and {chan_2}. Skipping...")
                    continue

            if params_exist:
                fig,ax = plt.subplots()
                frac_scatter(sim_1,sim_2,data_1,data_2,chan_1,chan_2,ax)

                if self.xlim is not None:
                    xlim = self.xlim if isinstance(self.xlim,tuple) else self.xlim[self.selected_shell3]
                    ax.set_xlim(*xlim)

                if self.add_random:
                    s = ''.join(random.choices(string.ascii_lowercase, k=6))
                fig.savefig(os.path.join(self.img_dir,f"{chan_1}_{chan_2}_scatter{s}{self.img_extension}"))

    def _redraw_static3(self):
        err_message = "No fit"
        rescaler = self._rescale if self.transform_setting == 'Forward' else self._inverse_rescale

        with self.output3:
            clear_output(wait=True)

            params_exist = True
            rescale = True

            chan_1, chan_2 = self.selected_chan3_1, self.selected_chan3_2
            shell = self.selected_shell3

            sim_1, sim_2 = self._get_frac('sim',shell,chan_1,chan_2)
            data_1, data_2 = self._get_frac('data',shell,chan_1,chan_2)

            if self.which_validation == 'Fitted':
                params_1 = self.fitted_params[chan_1]
                params_2 = self.fitted_params[chan_2]
                if params_1 == [] or params_2 == []:
                    params_exist = False
            elif self.which_validation == 'Initial':
                if self.init_params is None:
                    params_exist = False
                else:
                    params_1 = self.init_params.get(chan_1)
                    params_2 = self.init_params.get(chan_2)
                    if params_1 is None or params_2 is None:
                        params_exist = False
            elif self.which_validation == 'No rescale':
                rescale = False
            
            if rescale and params_exist:
                try:
                    sim_1, data_1, sim_2, data_2 = rescaler(sim_1,data_1,chan_1,self.which_validation,
                                                            sim_2,data_2,chan_2)
                except Exception as e:
                    params_exist = False
                    err_message = f"Cannot find inverse because of exception:\n{e}"
            
            if params_exist:
                num_bins = self._get_nbins(self.selected_shell3)
                self.fig3, (self.ax_hist3_1,
                            self.ax_hist3_2,
                            self.ax_hist3_3) = frac_plotter(sim_1,sim_2,
                                                            data_1,data_2,
                                                            chan_1,chan_2,
                                                            self.dpi,num_bins)
            else:
                self.fig3, (self.ax_hist3_1,
                            self.ax_hist3_2,
                            self.ax_hist3_3) = get_empty(self.dpi)
                
                for ax in [self.ax_hist3_1,self.ax_hist3_2,self.ax_hist3_3]:
                    ax.text(0.5,0.5,err_message,transform=ax.transAxes,ha='center',va='center')
            
            self._set_lims(self.ax_hist3_1,
                           self.ax_hist3_2,
                           self.ax_hist3_3,
                           self.selected_shell3)

            self.fig3.suptitle("Validation plots",fontsize=14)
            plt.show()
    
    
    def auto_fit(self,fit_dict=None,verbose=0):
        '''
        Automatically fits and updates the parameters

        :param fit_dict (dict or None): A dictionary of the form {(chan_1,chan_2):(0 or 1,0 or 1),...}
                                        The order of the dictionary corresponds to the order that the fitting 
                                        will be done
                                        If 0, fitting will not be done for that channel and it will use the 
                                        previous fitted parameters
                                        If 1, fitting will be done for that channel and the fitted parameters
                                        will be updated
                                        If none, fit_dict must have been specified during initialization
        :param verbose (int): Must be 0, 1, or 2
                              The larger the number, the more fitting information will be printed
        
        Once you are done doing the auto_fit, you may want to run .export_params(), which will export
        a .csv file containing the fitted parameters and uncertainties. The filepath was either
        specified during initialization with fitted_export_path or is 'params.csv'
        '''
        fit_dict = self.fit_dict if fit_dict is None else fit_dict
        self.channels_fitted_label = widgets.Label(value="Channels fitted:")
        self._clear(redraw=False)

        if fit_dict == {}:
            raise ValueError("No fit dictionary given!")

        self.output = nullcontext()

        def check_fit_dict(fit_dict):
            fit_counts = {c:0 for c in self.channels}
            counts = {c:0 for c in self.channels}
            
            for (chan1,chan2), (fit1,fit2) in fit_dict.items():
                counts[chan1] += 1
                counts[chan2] += 1

                if counts[chan1] == 1 and fit1 != 1:
                    raise ValueError(f"Issue with fit dict: {chan1} is used before it's fit!")
                    
                if counts[chan2] == 1 and fit2 != 1:
                    raise ValueError(f"Issue with fit dict: {chan2} is used before it's fit!")
                
                if fit1:
                    fit_counts[chan1] += 1
                
                if fit2:
                    fit_counts[chan2] += 1

            for chan, num in fit_counts.items():
                if num >= 2:
                    raise ValueError(f"Issue with fit dict: {chan} is fit more than one time!")
                elif num == 0:
                    raise ValueError(f"Issue with fit dict: {chan} is never fit!")

        check_fit_dict(fit_dict)

        for (chan1, chan2), (fit1, fit2) in fit_dict.items():
            if verbose == 1:
                extra1 = "float" if fit1 else "fixed"
                extra2 = "float" if fit2 else "fixed"

                print(f"Current fitting {chan1} ({extra1}) and {chan2} ({extra2})")

            for c in self.channels:
                self.enabled[c] = False
                
            self.enabled[chan1] = True
            self.enabled[chan2] = True
            
            self._fit(verbose=verbose,update_fit_dict=False)
        
        return self.fitted_params, self.errs

    def load_init_params(self,filepath):
        '''
        Loads the initial parameters from an exported .csv file
        The expected format is the same format gotten from pressing Export on 
        find_init_params

        :param filepath (str): The filepath of the .csv file
        '''
        df = pd.read_csv(filepath,index_col=0)

        self.init_params = {}

        for chan in df.index:
            params = []
            bounds = []

            i = 0
            while f"p{i}_0" in df.columns:
                val = df.loc[chan,f"p{i}_0"]

                if pd.isna(val) or val == "":
                    break

                params.append(float(val))

                bounds_col = f"p{i}_bounds"
                if bounds_col in df.columns:
                    bounds_str = df.loc[chan,bounds_col]

                    if isinstance(bounds_str,str) and bounds_str.strip():
                        bounds.append(tuple(ast.literal_eval(bounds_str)))

                i += 1
            
            self.init_params[chan] = params
            self.bounds[chan] = bounds
        
        self.n_params = {c:len(params) for c,params in self.init_params.items()}

    def load_all(self,filepath):
        '''
        Loads all the parameters and uncertainties from an exported .csv file
        The expected format is the same format gotten from pressing Export on
        semiauto_fit

        :param filepath (str): The filepath of the .csv file
        '''
        df = pd.read_csv(filepath,index_col=0)

        self.init_params = {}
        self.fitted_params = {}
        self.bounds = {}
        self.errs = {}

        for chan in df.index:
            init_params = []
            fitted_params = []
            bounds = []

            errs = {
                "parab":[],
                "+":[],
                "-":[],
                "gcc":[]
            }

            i = 0
            while f"p{i}_0" in df.columns:
                init_val = df.loc[chan,f"p{i}_0"]

                if not pd.isna(init_val) and init_val != "":
                    init_params.append(float(init_val))
                
                fit_val = df.loc[chan,f"p{i}"]
                if isinstance(fit_val,str) and fit_val.strip():
                    m = re.match(
                        r'^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
                        r'(?:\s*±([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?'
                        r'(?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/'
                        r'\+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?',
                        fit_val
                    )

                    if m:
                        fitted_params.append(float(m.group(1)))

                        if m.group(2):
                            errs["parab"].append(float(m.group(2)))

                        if m.group(3):
                            errs["-"].append(float(m.group(3)))

                        if m.group(4):
                            errs["+"].append(float(m.group(4)))
                        
                        errs["gcc"].append(np.nan)
                    
                bounds_col = f"p{i}_bounds"
                if bounds_col in df.columns:
                    bounds_str = df.loc[chan,bounds_col]

                    if isinstance(bounds_str,str) and bounds_str.strip():
                        bounds.append(tuple(ast.literal_eval(bounds_str)))
                
                i += 1

            self.init_params[chan] = init_params
            self.fitted_params[chan] = fitted_params
            self.errs[chan] = errs
            self.bounds[chan] = bounds
        
        self.n_params = {c:len(params) for c,params in self.init_params.items()}