#!/usr/bin/env python
# coding: utf-8

# # Make plots

# In[1]:


# # Import required modules

# In[3]:


import numpy as np, os, sys, glob, tools

import getdist, cobaya
from getdist import plots, MCSamples

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import argparse


from pylab import *
import tools

from scipy.stats import chi2

"""
which_ilc_sets = sys.argv[1]
cib_scatter_sigma = float(sys.argv[2]) ##0.2 ##None
if cib_scatter_sigma == -1:
    cib_scatter_sigma = None
tmpiter_key = sys.argv[3] ###'cibmindata_tsz' #'sim_tsz'
fit_for_cib_cal = int(sys.argv[4])
if fit_for_cib_cal:
    assert cib_scatter_sigma is None
#debug_cobaya = False #True ##False ##True
#force_resampling = True
"""

parser.add_argument('-which_ilc_sets', dest='which_ilc_sets', action='store', help='which_ilc_sets', type = str)
parser.add_argument('-cib_scatter_sigma', dest='cib_scatter_sigma', action='store', help='cib_scatter_sigma', type = float, default = None)
parser.add_argument('-tmpiter_key', dest='tmpiter_key', action='store', help='tmpiter_key', type = str, default = 'cibmindata_tsz')
parser.add_argument('-fit_for_cib_cal', dest='fit_for_cib_cal', action='store', help='fit_for_cib_cal', type = int, default = 0)
parser.add_argument('-debug_cobaya', dest='debug_cobaya', action='store', help='debug_cobaya', type = int, default = 0)
parser.add_argument('-force_resampling', dest='force_resampling', action='store', help='force_resampling', type = int, default = 1)


args = parser.parse_args()
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]
    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

if cib_scatter_sigma == -1 or cib_scatter_sigma == 'None':
    cib_scatter_sigma = None

assert which_ilc_sets in ['mv-cibfree', 'mv-mvcrosscibfree', 'cibfree-mvcrosscibfree']
totthreads = 10
os.putenv('OMP_NUM_THREADS',str(totthreads))

#fname = 'results/power_spectra_lmin500_lmax7000_deltal250/100d_tsz_final_estimate.npy'
#fname = 'results/power_spectra_lmin500_lmax7000_deltal250/100d_tsz_final_estimate_beamrc5.1_noslope.npy'
fname = 'results/power_spectra_lmin500_lmax5000_deltal500/100d_tsz_final_estimate_beamrc5.1_noslope.npy'
res_dic = np.load(fname, allow_pickle=True).item()
##print(res_dic.keys()); sys.exit()
op_ps_1d_dic = res_dic['op_ps_1d_dic']
full_stat_cov = res_dic['full_stat_cov']
full_sys_cov_dic = res_dic['full_sys_cov_dic']
m1m2_arr = [('ymv', 'ymv'), ('ycibfree', 'ycibfree'), ('ycibfree', 'ymv')]
tmpels = res_dic[m1m2_arr[0]]['els']

final_full_sys_cov = full_sys_cov_dic['cib_tweaked_spt_only_max_tweak_0.2'] + \
                full_sys_cov_dic['rad_tweaked_max_tweak_0.2'] + \
                full_sys_cov_dic['cmb_withspiretcalerror'] + \
                full_sys_cov_dic['ksz']
            
full_cov = full_stat_cov + final_full_sys_cov

#sim tszxCIB estimates
sim_tsz_cib_estimate_fname = 'results/power_spectra_lmin500_lmax5000_deltal500/100d_tsz_final_estimate_beamrc5.1_noslope_sim_tszcib_estimates.npy'
sim_tsz_cib_estimate_dic = np.load( sim_tsz_cib_estimate_fname, allow_pickle=True).item()
#print(sim_tsz_cib_estimate_dic.keys()); sys.exit()

total_sims_for_tsz_cib = 50
sim_ps_dic = sim_tsz_cib_estimate_dic['sim_ps_dic']
bands = sim_tsz_cib_estimate_dic['bands']
ilc_1d_weights_dic = sim_tsz_cib_estimate_dic['ilc_1d_weights_dic']
#cl_yy_fromcibmindata = sim_tsz_cib_estimate_dic['cl_yy_fromcibmindata']

wl_dic = {}
for ilc_keyname in ['ymv', 'ycibfree']:
    wl_arr = []
    els = ilc_1d_weights_dic[ilc_keyname]['els']
    for freq in ilc_1d_weights_dic[ilc_keyname]:
        if freq == 'els': continue
        binned_weights = ilc_1d_weights_dic[ilc_keyname][freq]                
        wl = np.interp( tmpels, els, binned_weights )
        wl_arr.append( wl )
    wl_arr = np.asarray(wl_arr)
    wl_dic[ilc_keyname] = wl_arr


#full_stat_corr = corr_from_cov(full_stat_cov)
##imshow(full_stat_corr); colorbar(); show()
linds = np.arange(len(tmpels))
ilc_keyname1, ilc_keyname2, ilc_keyname3 = ('ymv', 'ymv'), ('ycibfree', 'ycibfree'), ('ycibfree', 'ymv')
tmpels = tmpels[linds]
curr_dl_fac = tmpels * (tmpels+1)/2/np.pi * 1e12
d1_undesired_comp = res_dic[ilc_keyname1]['sim'][linds]
d2_undesired_comp = res_dic[ilc_keyname2]['sim'][linds]
d3_undesired_comp = res_dic[ilc_keyname3]['sim'][linds]
d1 = res_dic[ilc_keyname1]['data_final'][linds]
d2 = res_dic[ilc_keyname2]['data_final'][linds]
d3 = res_dic[ilc_keyname3]['data_final'][linds]
reclen = len(d1)

print('\n\n\n')
#difference tests
which_sim = 'cmb_tsz_ksz_noise_uncorrcib_uncorrrad_rc5.1_noslope_spt3gbeams_compdependent_half_splits'

#lmin_lmax_arr = [(500, 3000)]
#lmin_lmax_arr = [(500, 1500), (1500, 3000)]
#lmin_lmax_arr = [(500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000)]
lmin_lmax_arr = [(500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, 5000)]
#lmin_lmax_arr = [(500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, 3500), (3500, 4000), (4000, 4500), (4500, 5000)]

if which_ilc_sets == 'mv-cibfree':
    undesired_comp_for_sima, undesired_comp_for_simb = d1_undesired_comp, d2_undesired_comp
    curr_diff_vector = d1-d2
    key_for_sima, key_for_simb = ilc_keyname1, ilc_keyname2
elif which_ilc_sets == 'mv-mvcrosscibfree':
    undesired_comp_for_sima, undesired_comp_for_simb = d1_undesired_comp, d3_undesired_comp
    curr_diff_vector = d1-d3
    key_for_sima, key_for_simb = ilc_keyname1, ilc_keyname3
elif which_ilc_sets == 'cibfree-mvcrosscibfree':
    undesired_comp_for_sima, undesired_comp_for_simb = d2_undesired_comp, d3_undesired_comp
    curr_diff_vector = d2-d3
    key_for_sima, key_for_simb = ilc_keyname2, ilc_keyname3
#-----------------------------------
import cobaya

curr_full_cov = tools.get_covs_for_difference_vectors(tmpels, key_for_sima, key_for_simb, op_ps_1d_dic, lmin_lmax_arr = lmin_lmax_arr)
##print(curr_full_cov.shape); sys.exit()
tmpreclen = int( len(curr_full_cov)/2 )
c_aa = curr_full_cov[:tmpreclen, :tmpreclen]
c_bb = curr_full_cov[tmpreclen:, tmpreclen:]
c_ab = curr_full_cov[tmpreclen:, :tmpreclen]
curr_diff_cov = c_aa + c_bb - 2*c_ab
##imshow(curr_diff_cov); colorbar(); show(); sys.exit()
##print(curr_diff_cov.shape); sys.exit()

def get_data_vectors(lmin_lmax_arr):
    data = []
    for bincntr, l1l2 in enumerate( lmin_lmax_arr ):
        l1, l2 = l1l2
        curr_rho_tsz_cib_linds = np.where( (tmpels>=l1) & (tmpels<l2) )[0]

        #fitting
        curr_data = curr_diff_vector[curr_rho_tsz_cib_linds]
        data.extend( curr_data )

    return np.asarray( data )

def get_model_vectors(lmin_lmax_arr, param_dict_sampler, sim_or_data_tsz = 'cibmindata_tsz', cib_scatter_sigma = None):
    model = []
    for bincntr, l1l2 in enumerate( lmin_lmax_arr ):
        ##print(l1l2)
        l1, l2 = l1l2
        if param_dict_sampler is not None:
            ppp_name = 'rho_tsz_cib_%s' %(bincntr+1)
            curr_rho_tsz_cib = param_dict_sampler[ppp_name]
            cib_cal_1 = param_dict_sampler['cib_cal_1']
            cib_cal_2 = param_dict_sampler['cib_cal_2']
            cib_cal_3 = param_dict_sampler['cib_cal_3']
            cib_cal_4 = param_dict_sampler['cib_cal_4']
            cib_cal_5 = param_dict_sampler['cib_cal_5']
            cib_cal_6 = param_dict_sampler['cib_cal_6']
        else:
            curr_rho_tsz_cib = None

        curr_rho_tsz_cib_linds = np.where( (tmpels>=l1) & (tmpels<l2) )[0]
        
        sa_arr = tools.get_sim_arrary(res_dic, key_for_sima, which_sim) - undesired_comp_for_sima
        sb_arr = tools.get_sim_arrary(res_dic, key_for_simb, which_sim) - undesired_comp_for_simb
        
        sa_arr, sb_arr = tools.account_for_tsz_cib_in_sims(curr_rho_tsz_cib, sa_arr, sb_arr, 
                                                           sim_ps_dic, 
                                                           bands, 
                                                           wl_dic, 
                                                           key_for_sima, key_for_simb,
                                                           sim_tsz_cib_estimate_dic,
                                                           total_sims_for_tsz_cib = total_sims_for_tsz_cib, 
                                                           sim_or_data_tsz = sim_or_data_tsz,
                                                           reqd_linds = curr_rho_tsz_cib_linds, 
                                                           cib_scatter_sigma = cib_scatter_sigma, 
                                                           cib_cal_1 = cib_cal_1,
                                                           cib_cal_2 = cib_cal_2,
                                                           cib_cal_3 = cib_cal_3,
                                                           cib_cal_4 = cib_cal_4,
                                                           cib_cal_5 = cib_cal_5,
                                                           cib_cal_6 = cib_cal_6,
                                                          )

        curr_diff_vector_sim_arr = sa_arr - sb_arr
        curr_diff_vector_sim_arr = curr_diff_vector_sim_arr[25:]

        #fitting
        curr_model = np.mean( curr_diff_vector_sim_arr, axis = 0)[curr_rho_tsz_cib_linds]
        model.extend( curr_model )
            
    return np.asarray( model )


def get_tsz_cib_corr_likelihood(**param_values):
    import copy
    param_values = [param_values[p] for p in param_names]
    param_dict_sampler = {}
    for pcntr, ppp in enumerate( param_names ):
        param_dict_sampler[ppp] = param_values[pcntr]
    
    model = get_model_vectors(lmin_lmax_arr, param_dict_sampler, sim_or_data_tsz = tmpiter_key, cib_scatter_sigma = cib_scatter_sigma)
    ##print(data.shape, model.shape, curr_diff_cov.shape); sys.exit()
    

    res = tools.get_likelihood(data, model, curr_diff_cov)
    return res

total_bins = len( lmin_lmax_arr )
data = get_data_vectors(lmin_lmax_arr)

rho_tsz_cib_ref_dict = {
                "prior": {"min": -1., "max": 1.},
                "ref": {"dist": "norm", "loc": 0.2, "scale": 0.2},
                "proposal": 0.2,
                "drop": False, 
                "latex": r"\rho_{\rm tSZxCIB_binval}", 
                }
if fit_for_cib_cal:
    cib_ref_dict = {
                    "prior": {"min": 0.7, "max": 1.3},
                    "ref": {"dist": "norm", "loc": 1., "scale": 0.3},
                    "proposal": 0.2,
                    "drop": False, 
                    "latex": r"CIB^{\rm Cal_{bandcntrval}}", 
                    }

mcmc_input_params_info_dict = {}
for binno in range(total_bins):
    paramname = 'rho_tsz_cib_%s' %(binno+1)
    mcmc_input_params_info_dict[paramname] = {}
    for keyname in rho_tsz_cib_ref_dict:
        mcmc_input_params_info_dict[paramname][keyname] = rho_tsz_cib_ref_dict[keyname]
        if keyname == 'latex':
            currval = mcmc_input_params_info_dict[paramname][keyname]
            mcmc_input_params_info_dict[paramname][keyname] = currval.replace('binval', '%s' %(binno+1))
if fit_for_cib_cal:
    for bandcntr, bandval in enumerate( bands ):
        paramname = 'cib_cal_%s' %(bandcntr+1)
        mcmc_input_params_info_dict[paramname] = {}
        for keyname in cib_ref_dict:
            mcmc_input_params_info_dict[paramname][keyname] = cib_ref_dict[keyname]
            if keyname == 'latex':
                currval = mcmc_input_params_info_dict[paramname][keyname]
                mcmc_input_params_info_dict[paramname][keyname] = currval.replace('bandcntrval', '%s' %(bandcntr+1))
    ###print(mcmc_input_params_info_dict); sys.exit()


#debug_cobaya = False #True ##False ##True
#force_resampling = True
GRstat = 0.01
#chain_name = 'tsz_cib_corr_samples_%sbins' %(total_bins)
lmin_lmax_arr_str = 'lbins'
for l1l2 in lmin_lmax_arr:
    l1, l2 = l1l2
    l1l2_str = '%sto%s' %(l1, l2)
    lmin_lmax_arr_str = '%s-%s' %(lmin_lmax_arr_str, l1l2_str)
chain_name = 'tszcibcorr_%s_totalbins%s_%s' %(which_ilc_sets, total_bins, lmin_lmax_arr_str)
if fit_for_cib_cal:
    chain_fd_and_name = 'results/chains/%s/fit_for_cib_cal/%s/%s' %(tmpiter_key, chain_name, chain_name)
else:
    chain_fd_and_name = 'results/chains/%s/cib_scatter_sigma_%s/%s/%s' %(tmpiter_key, cib_scatter_sigma, chain_name, chain_name)

input_info = {}
input_info["params"] = mcmc_input_params_info_dict


param_names = list(input_info["params"].keys())
#print( param_names ); ###sys.exit()

input_info["likelihood"] = {}
input_info["likelihood"]["tsz_cib_fitting"] = {
                "external": get_tsz_cib_corr_likelihood, 
                "input_params": param_names,
                }

input_info["sampler"] = {"mcmc": {"drag": False, "Rminus1_stop": GRstat, "max_tries": 5000}}

input_info["output"] = chain_fd_and_name
updated_info, sampler = cobaya.run(input_info, force = force_resampling, debug = debug_cobaya)
print('Done.')





