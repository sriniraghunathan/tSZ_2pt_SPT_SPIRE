import numpy as np, sys, os
from pylab import *

def get_constraints_table(params_to_plot, sample_arr_to_plot, color_arr = None, rounding_dic = {}):
    if color_arr is None:
        color_arr = np.tile(None, len( sample_arr_to_plot) )
    #get the constraints
    nx, ny = len(sample_arr_to_plot), len( params_to_plot )
    constraints_dic = {}
    constraints_table = np.empty( (nx, ny), dtype = '<U30' )
    colors_table = np.empty( (nx, ny), dtype = '<U30' )
    col_labels = []
    for pind, ppp in enumerate( params_to_plot ):
        constraints_dic[ppp] = {}
        for sind, (s, c) in enumerate( zip( sample_arr_to_plot, color_arr) ):
            ###print( ppp, s.getLatex(ppp) )
            #tmp = s.getLatex(ppp)
            tmp = s.getInlineLatex(ppp)#, limit = 1, err_sig_figs = 3)
            if tmp.find('<')>-1: #increase to 95% C.L>
                tmp = s.getInlineLatex(ppp, limit = 2)#, err_sig_figs = 3)
                if tmp.find('--')>-1:
                    tmp = tmp.replace('---', '-')
                else:
                    tmp = r'%s\ ({\rm 95\%% C.L.})' %(tmp)
            else:
                if ppp in rounding_dic:
                    errval = float(tmp.split('pm')[1])
                    errval_rounded = round(errval, rounding_dic[ppp])
                    tmp = tmp.replace('%g' %(errval), '%s' %(errval_rounded))
            param_label, curr_val = strip_getdist_latex_str(tmp)
            constraints_table[sind, pind] = r'$%s$' %(curr_val)
            colors_table[sind, pind] = c
            constraints_dic[ppp][sind] = r'$%s$' %(curr_val)
        col_labels.append( r'$\sigma(%s)$' %(param_label) )
    return constraints_dic, constraints_table, colors_table, col_labels

def write_errors_in_diagonal_posteriors(g, params_to_plot, color_arr, constraints_dic, legfsval = 10, legloc = 4, handlelength = 1, handletextpad = 0.3, ncol = 2, frameon = True, labelspacing = 0.3, lwval = 0.7):
    total_subplots = len( g.subplots )

    if isinstance(legloc, int):
        legloc_arr = np.tile(legloc, total_subplots)
    else:
        legloc_arr = legloc

    for r in range( total_subplots ):
        for c in range( total_subplots ):
            if c!=r: continue
            ax = g.subplots[r,c]
            ppp = params_to_plot[c]
            for sampleind in constraints_dic[ppp]:
                curr_val = constraints_dic[ppp][sampleind]
                ax.plot([], [], color = color_arr[sampleind], label = curr_val, lw = lwval)
            
            handles, labels = ax.get_legend_handles_labels()
            leg=ax.legend(handles[sampleind+1:], labels[sampleind+1:], loc = legloc_arr[c], fontsize = legfsval, handlelength = handlelength, handletextpad = handletextpad, ncol = ncol, frameon=frameon, labelspacing = labelspacing)
            leg.get_frame().set_linewidth(0.0)
            #ax.legend(loc = 4, fontsize = legfsval)
    return g   

def strip_getdist_latex_str(later_str, what_kind_of_constraint = 'full'):
    if later_str.find('{')>-1:
        delimiters_for_stripping = ['$', '^{+', '}_{', '}$' ]
    else:
        delimiters_for_stripping = ['$', '\\pm', '$']
    regex_for_stripping = '|'.join(map(re.escape, delimiters_for_stripping))
    curr_val_split = re.split(regex_for_stripping, later_str)
    curr_val_split = [c for c in curr_val_split if c.strip()]
    param_label = None
    if what_kind_of_constraint == 'full':
        if later_str.find('=')>-1:
            param_label, curr_val = later_str.split('=')
        elif later_str.find('<')>-1:
            param_label, curr_val = later_str.split('<')
            curr_val = '<%s' %(curr_val)
        elif later_str.find('>')>-1:
            param_label, curr_val = later_str.split('>')
            curr_val = '>%s' %(curr_val)
        #print(param_label, curr_val)
        curr_val = curr_val.strip()
    elif what_kind_of_constraint == 'upper_error':
        curr_val = curr_val_split[1]
    elif what_kind_of_constraint == 'lower_error':
        if len(curr_val_split)==2:
            curr_val = curr_val_split[1]
        elif len(curr_val_split) == 3:
            curr_val = curr_val_split[2]
    elif what_kind_of_constraint == 'best_fit':
        curr_val = curr_val_split[0]

    if what_kind_of_constraint != 'full':
        ##print(curr_val_split, curr_val)
        if curr_val.find('-')>-1:
            curr_val = float( curr_val.strip('-') ) * -1
        else:
            curr_val = float( curr_val )
    if curr_val in ['--', '---', '_', '-']:
        curr_val = r'{\rm N/A}'

    return param_label, curr_val

def param_mapping(p):
    param_map_dic = {'w': 'ws', 
                    'wa': 'wa', 
                    'nnu': 'neff', 
                    }
    if p in param_map_dic:
        return param_map_dic[p]
    else:
        return p

def mark_axlines(g, params_to_plot, param_dict = None, lwval = 0.5, lsval = '-', alphaval = 0.5, zorderval = 10):
    total_subplots = len( g.subplots )

    for r in range( total_subplots ):
        for c in range( total_subplots ):
            if c>r:continue
            ax = g.subplots[r,c]
            ax.tick_params('both', length=2, width=0.5, which='major', direction = 'in')
            ax.tick_params('both', length=1, width=0.5, which='minor', direction = 'in')

            if total_subplots>1:
                p1, p2 = param_mapping(params_to_plot[c]), param_mapping(params_to_plot[r])
            else:
                p1, p2 = params_to_plot[0]
                p1, p2 = param_mapping(p1), param_mapping(p2)
            ##print(p1, p2)
            if r == c: #diagonal
                if total_subplots>1:
                    ax.yaxis.set_visible(False)
                    if param_dict is not None and p1 in param_dict:
                        pval = param_dict[p1]
                        ax.axvline(pval, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval); 
                else:
                    if param_dict is not None and p1 in param_dict:
                        p1val = param_dict[p1]
                        ax.axvline(p1val, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval)
                    if param_dict is not None and p2 in param_dict:
                        p2val = param_dict[p2]
                        ax.axhline(p2val, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval)
            else:
                if param_dict is not None and p1 in param_dict:
                    p1val = param_dict[p1]
                    ax.axvline(p1val, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval)
                if param_dict is not None and p2 in param_dict:
                    p2val = param_dict[p2]
                    ax.axhline(p2val, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval)
    return g

def get_param_cov_mat_from_getdist(samples, params):
    cov_mat = samples.cov(params)
    return cov_mat

def get_modified_cov(cl11_arr, cl22_arr, operation):

    total_sims = len(cl11_arr)
    assert total_sims == len(cl22_arr)
    cl_arr_for_cov = []
    for i in range( total_sims ):
        if operation == 'subtract':
            curr_cl_for_cov = cl11_arr[i] - cl22_arr[i]
        elif operation == 'divide':
            curr_cl_for_cov = cl11_arr[i] / cl22_arr[i]
            #print(cl11_arr[i], cl22_arr[i]); sys.exit()
        elif operation == 'multiply':
            curr_cl_for_cov = cl11_arr[i] * cl22_arr[i]
        elif operation == 'add':
            curr_cl_for_cov = cl11_arr[i] + cl22_arr[i]

        cl_arr_for_cov.append( curr_cl_for_cov )

    cl_arr_for_cov = np.asarray( cl_arr_for_cov )
    cov = np.cov(cl_arr_for_cov.T)
    
    return cov

def get_covs_for_difference_vectors(binned_el, m1_comb, m2_comb, 
            op_ps_1d_dic, 
            sim_comp_for_non_gau_errror = 'cmb_tsz_ksz_noise_uncorrcib_uncorrrad', 
            total_sims = 100,
            sys_comp_arr = ['cib_tweaked',
                             'rad_tweaked',
                             'cib_tweaked_spt_only',
                             'cib_tweaked_spt_only_max_tweak_0.2',
                             'cib_tweaked_max_tweak_0.2',
                             'rad_tweaked_max_tweak_0.2',],
            comp_arr_for_final_cov = ['stat', 'cib_tweaked_spt_only_max_tweak_0.2', 'rad_tweaked_max_tweak_0.2', 'cmb_withspiretcalerror', 'ksz'],
            lmin_lmax_arr = [[0, 5001]],
            ):

    #Do the difference between two sets of sims and compute the covariance

    """
    m1_comb: ('ymv', 'ymv') or something like that.
    m2_comb: ('ymv', 'ymv') or something like that.
    sim_comp_for_non_gau_errror: "Full" sim array for cov estimation
    """

    def arrange_and_concat_cls(lmin_lmax_arr, cl_1, cl_2):
        """
        cl_concat = []
        for lmin_lmax in lmin_lmax_arr:
            lmin, lmax = lmin_lmax
            linds = np.where( (binned_el>=lmin) & (binned_el<lmax) )[0]
            cl_concat.extend( np.concatenate( (cl_1[linds], cl_2[linds]) ) )
        """
        cl_concat1, cl_concat2 = [], []
        for lmin_lmax in lmin_lmax_arr:
            lmin, lmax = lmin_lmax
            linds = np.where( (binned_el>=lmin) & (binned_el<lmax) )[0]
            cl_concat1.extend( cl_1[linds] )
            cl_concat2.extend( cl_2[linds] )
            ##print(cl_concat1)
            ##print(cl_concat2); sys.exit()
        cl_concat = np.concatenate( (cl_concat1, cl_concat2) )
        return cl_concat

    cov_dic = {}
    cl_full_arr_for_cov = []
    for simno in sorted( op_ps_1d_dic[sim_comp_for_non_gau_errror] ):
        #stat
        el_, cl_1 = op_ps_1d_dic[sim_comp_for_non_gau_errror][simno][0][m1_comb]
        el_, cl_2 = op_ps_1d_dic[sim_comp_for_non_gau_errror][simno][0][m2_comb]

        cl_concat = arrange_and_concat_cls(lmin_lmax_arr, cl_1, cl_2)
        ##print(cl_concat); sys.exit()
        cl_full_arr_for_cov.append( cl_concat )
    cl_full_arr_for_cov = np.asarray( cl_full_arr_for_cov )
    cov_dic['stat'] = np.cov(cl_full_arr_for_cov.T)


    #systematics
    for sys_comp in sys_comp_arr:
        cl_full_arr_for_cov = []
        for simno in sorted( op_ps_1d_dic[sys_comp] ):
            el_, cl_1 = op_ps_1d_dic[sys_comp][simno][0][m1_comb]
            el_, cl_2 = op_ps_1d_dic[sys_comp][simno][0][m2_comb]
            cl_concat = arrange_and_concat_cls(lmin_lmax_arr, cl_1, cl_2)
            ##print(cl_concat); sys.exit()
            cl_full_arr_for_cov.append( cl_concat )
        cl_full_arr_for_cov = np.asarray( cl_full_arr_for_cov )
        
        cov_dic[sys_comp] = np.cov(cl_full_arr_for_cov.T)

    #CMB systematic with Tcal error and kSZ systematic
    m1_comb_rev, m2_comb_rev = m1_comb[::-1], m2_comb[::-1]
    if m1_comb in op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter']:
        cl_cmb_ilc_res_arr_1 = np.asarray( op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter'][m1_comb][:total_sims] ) / 1e12
    else:
        cl_cmb_ilc_res_arr_1 = np.asarray( op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter'][m1_comb_rev][:total_sims] ) / 1e12
    if m2_comb in op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter']:
        cl_cmb_ilc_res_arr_2 = np.asarray( op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter'][m2_comb][:total_sims] ) / 1e12
    else:
        cl_cmb_ilc_res_arr_2 = np.asarray( op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter'][m2_comb_rev][:total_sims] ) / 1e12
    
    cl_full_arr_for_cov = []
    for simno in range( total_sims ):
        cl_1, cl_2 = cl_cmb_ilc_res_arr_1[simno], cl_cmb_ilc_res_arr_2[simno]
        cl_concat = arrange_and_concat_cls(lmin_lmax_arr, cl_1, cl_2)
        """
        cl_concat = []
        for lmin_lmax in lmin_lmax_arr:
            lmin, lmax = lmin_lmax
            linds = np.where( (binned_el>=lmin) & (binned_el<lmax) )[0]
            cl_concat.extend( np.concatenate( (cl_1[linds], cl_2[linds]) ) )
        """
        #print(cl_concat); sys.exit()
        cl_full_arr_for_cov.append( cl_concat )
    cl_full_arr_for_cov = np.asarray( cl_full_arr_for_cov )
    cov_dic['cmb_withspiretcalerror'] = np.cov(cl_full_arr_for_cov.T)

    #kSZ
    if m1_comb in op_ps_1d_dic['ksz_cl_ilc_res_dict']:
        cl_ksz_ilc_res_arr_1 = np.asarray( list( op_ps_1d_dic['ksz_cl_ilc_res_dict'][m1_comb].values() ) ) / 1e12
    else:
        cl_ksz_ilc_res_arr_1 = np.asarray( list( op_ps_1d_dic['ksz_cl_ilc_res_dict'][m1_comb_rev].values() ) ) / 1e12
    if m2_comb in op_ps_1d_dic['ksz_cl_ilc_res_dict']:
        cl_ksz_ilc_res_arr_2 = np.asarray( list( op_ps_1d_dic['ksz_cl_ilc_res_dict'][m2_comb].values() ) ) / 1e12
    else:
        cl_ksz_ilc_res_arr_2 = np.asarray( list( op_ps_1d_dic['ksz_cl_ilc_res_dict'][m2_comb_rev].values() ) ) / 1e12

    cl_full_arr_for_cov = []
    for simno in range( total_sims ):
        cl_1, cl_2 = cl_ksz_ilc_res_arr_1[simno], cl_ksz_ilc_res_arr_2[simno]
        cl_concat = arrange_and_concat_cls(lmin_lmax_arr, cl_1, cl_2)
        """
        cl_concat = []
        for lmin_lmax in lmin_lmax_arr:
            lmin, lmax = lmin_lmax
            linds = np.where( (binned_el>=lmin) & (binned_el<lmax) )[0]
            cl_concat.extend( np.concatenate( (cl_1[linds], cl_2[linds]) ) )
        """
        #print(cl_concat); sys.exit()
        cl_full_arr_for_cov.append( cl_concat )
    cl_full_arr_for_cov = np.asarray( cl_full_arr_for_cov )
    cov_dic['ksz'] = np.cov(cl_full_arr_for_cov.T)

    final_cov = np.zeros_like( cov_dic['stat'] )
    for comp in comp_arr_for_final_cov:
        ##print(comp, np.mean(cov_dic[comp]))
        final_cov = final_cov + cov_dic[comp]
        ##print(final_cov); sys.exit()

    #print(final_cov); sys.exit()

    return final_cov

'''
def get_covs_for_difference_vectors_v1(binned_el, m1_comb, m2_comb, 
            op_ps_1d_dic, 
            sim_comp_for_non_gau_errror = 'cmb_tsz_ksz_noise_uncorrcib_uncorrrad', 
            total_sims = 100,
            sys_comp_arr = ['cib_tweaked',
                             'rad_tweaked',
                             'cib_tweaked_spt_only',
                             'cib_tweaked_spt_only_max_tweak_0.2',
                             'cib_tweaked_max_tweak_0.2',
                             'rad_tweaked_max_tweak_0.2',],
            comp_arr_for_final_cov = ['stat', 'cib_tweaked_spt_only_max_tweak_0.2', 'rad_tweaked_max_tweak_0.2', 'cmb_withspiretcalerror', 'ksz'],
            lmin_lmax_arr = [0, 50001],
            ):

    #Do the difference between two sets of sims and compute the covariance

    """
    m1_comb: ('ymv', 'ymv') or something like that.
    m2_comb: ('ymv', 'ymv') or something like that.
    sim_comp_for_non_gau_errror: "Full" sim array for cov estimation
    """
    lmin, lmax = lmin_lmax_arr[0]
    linds = np.where( (binned_el>=lmin) & (binned_el<lmax) )[0]
    print(binned_el[linds]); 
    return None
    cov_dic = {}
    cl_full_arr_for_cov = []
    for simno in sorted( op_ps_1d_dic[sim_comp_for_non_gau_errror] ):
        #stat
        el_, cl_1 = op_ps_1d_dic[sim_comp_for_non_gau_errror][simno][0][m1_comb]
        el_, cl_2 = op_ps_1d_dic[sim_comp_for_non_gau_errror][simno][0][m2_comb]

        cl_concat = np.concatenate( (cl_1[linds], cl_2[linds]) )
        cl_full_arr_for_cov.append( cl_concat )
    cl_full_arr_for_cov = np.asarray( cl_full_arr_for_cov )
    cov_dic['stat'] = np.cov(cl_full_arr_for_cov.T)


    #systematics
    for sys_comp in sys_comp_arr:
        cl_full_arr_for_cov = []
        for simno in sorted( op_ps_1d_dic[sys_comp] ):
            el_, cl_1 = op_ps_1d_dic[sys_comp][simno][0][m1_comb]
            el_, cl_2 = op_ps_1d_dic[sys_comp][simno][0][m2_comb]
            cl_concat = np.concatenate( (cl_1[linds], cl_2[linds]) )
            cl_full_arr_for_cov.append( cl_concat )
        cl_full_arr_for_cov = np.asarray( cl_full_arr_for_cov )
        cov_dic[sys_comp] = np.cov(cl_full_arr_for_cov.T)

    #CMB systematic with Tcal error and kSZ systematic
    m1_comb_rev, m2_comb_rev = m1_comb[::-1], m2_comb[::-1]
    if m1_comb in op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter']:
        cl_cmb_ilc_res_arr_1 = np.asarray( op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter'][m1_comb][:total_sims] ) / 1e12
    else:
        cl_cmb_ilc_res_arr_1 = np.asarray( op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter'][m1_comb_rev][:total_sims] ) / 1e12
    if m2_comb in op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter']:
        cl_cmb_ilc_res_arr_2 = np.asarray( op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter'][m2_comb][:total_sims] ) / 1e12
    else:
        cl_cmb_ilc_res_arr_2 = np.asarray( op_ps_1d_dic['cmb_cl_ilc_res_dict_withspiretcalscatter'][m2_comb_rev][:total_sims] ) / 1e12
    
    cl_full_arr_for_cov = []
    for simno in range( total_sims ):
        cl_concat = np.concatenate( (cl_cmb_ilc_res_arr_1[simno][linds], cl_cmb_ilc_res_arr_2[simno][linds]) )
        cl_full_arr_for_cov.append( cl_concat )
    cl_full_arr_for_cov = np.asarray( cl_full_arr_for_cov )
    cov_dic['cmb_withspiretcalerror'] = np.cov(cl_full_arr_for_cov.T)

    if m1_comb in op_ps_1d_dic['ksz_cl_ilc_res_dict']:
        cl_ksz_ilc_res_arr_1 = np.asarray( list( op_ps_1d_dic['ksz_cl_ilc_res_dict'][m1_comb].values() ) ) / 1e12
    else:
        cl_ksz_ilc_res_arr_1 = np.asarray( list( op_ps_1d_dic['ksz_cl_ilc_res_dict'][m1_comb_rev].values() ) ) / 1e12
    if m2_comb in op_ps_1d_dic['ksz_cl_ilc_res_dict']:
        cl_ksz_ilc_res_arr_2 = np.asarray( list( op_ps_1d_dic['ksz_cl_ilc_res_dict'][m2_comb].values() ) ) / 1e12
    else:
        cl_ksz_ilc_res_arr_2 = np.asarray( list( op_ps_1d_dic['ksz_cl_ilc_res_dict'][m2_comb_rev].values() ) ) / 1e12
    cl_full_arr_for_cov = []
    for simno in range( total_sims ):
        cl_concat = np.concatenate( (cl_ksz_ilc_res_arr_1[simno][linds], cl_ksz_ilc_res_arr_2[simno][linds]) )
        cl_full_arr_for_cov.append( cl_concat )
    cl_full_arr_for_cov = np.asarray( cl_full_arr_for_cov )
    cov_dic['ksz'] = np.cov(cl_full_arr_for_cov.T)

    final_cov = np.zeros_like( cov_dic['stat'] )
    for comp in comp_arr_for_final_cov:
        final_cov = final_cov + cov_dic[comp]

    return final_cov    
'''

def get_spt3g_rc5p1_noslope_beam(nu, mapparams = None, component = 'cmb', fpath = 'data/beams/rc5.1_noslope/products/componentval/B_ell.npz', lmax = -1, return_ell = False, beam_sigma = 0.):

    component_dic_for_rc5_beams = {'cmb': 'cmb', 'cib': 'cib', 'tsz': 'tsz', 'radio': 'rg', 'rad': 'rg', 'ksz': 'cmb'}

    fpath = fpath.replace('componentval', component_dic_for_rc5_beams[component])
    if not os.path.exists(fpath):
        fpath = '/home/sri//analysis/2020_07/ksz_ps/maps/mapmaking/3G/%s' %(fpath)

    bl_dic = np.load( fpath )
    
    el, bl = bl_dic['ell'], bl_dic[str(nu)]
    if lmax > 0:
        el = el[:lmax]
        bl = bl[:lmax]
    
    if beam_sigma != 0: #20251230
        fpath_cov = fpath.replace('B_ell', 'cov')
        bl_cov_dic = np.load(fpath_cov)
        bl_cov_el = bl_cov_dic['ell']
        bl_cov_bands = bl_cov_dic['bands'].astype(int)
        bl_cov = bl_cov_dic['cov'].real

        #extract cov/sigma for the respective band
        nuind = np.where( bl_cov_bands == nu )[0]
        cov_len = len( bl_cov_el )
        start = int( cov_len * nuind )
        end = start + cov_len
        bl_cov = bl_cov[start:end, start: end]

        bl_sigma = np.sqrt( np.diag(bl_cov) )
        bl_sigma = np.interp(el, bl_cov_el, bl_sigma)

        if (0):
            clf()
            errorbar(el, bl, yerr = bl_sigma)
            show(); sys.exit()

        #add sigma to bl now
        bl = bl + bl_sigma * beam_sigma


    if mapparams is not None: #convert to 2D
        bl = flatsky.cl_to_cl2d(el, bl, mapparams) 

    if return_ell:
        return el, bl
    else:
        return bl

def account_for_tsz_cib_in_sims(rho_tsz_cib, sa_arr, sb_arr, sim_ps_dic, bands, wl_dic, m1, m2, sim_tsz_cib_estimate_dic, total_sims_for_tsz_cib = 50, 
    sim_or_data_tsz = 'cibmindata_tsz',
    reqd_linds = None, 
    cib_scatter_sigma = None,
    cib_cal_1 = None,
    cib_cal_2 = None,
    cib_cal_3 = None,
    cib_cal_4 = None,
    cib_cal_5 = None,
    cib_cal_6 = None,
    # uncorr_cib_frac_a = None,
    # uncorr_cib_frac_b = None,
    rs=111,
    include_beam_chromaticity = 0,
    return_tsz_cib_estimates = False,
    res_tsz_cib_a_arr = None,
    res_tsz_cib_b_arr = None,
    explicitly_null_tsz_cib_in_cibfree = 0,
    ):
    if include_beam_chromaticity: #20251230 - account for beam chromaticity (i.e:) variation of beams for different SEDs
        bl_chrom_dic_for_cib, bl_chrom_dic_for_tsz = {}, {}
        for curr_band in bands:
            if curr_band in [90, 150, 220]:
                curr_bl_cmb = get_spt3g_rc5p1_noslope_beam(curr_band, component = 'cmb')
                curr_bl_cib = get_spt3g_rc5p1_noslope_beam(curr_band, component = 'cib')
                curr_bl_tsz = get_spt3g_rc5p1_noslope_beam(curr_band, component = 'tsz')
                curr_bl_chrom_dic_cib = curr_bl_cmb / curr_bl_cib
                curr_bl_chrom_dic_tsz = curr_bl_cmb / curr_bl_tsz
            else:
                curr_bl_chrom_dic_cib = None
                curr_bl_chrom_dic_tsz = None
            bl_chrom_dic_for_cib[curr_band] = curr_bl_chrom_dic_cib
            bl_chrom_dic_for_tsz[curr_band] = curr_bl_chrom_dic_tsz
        if (0):
            close('all')
            clf()
            color_dic = {90: 'navy', 150: 'darkgreen', 220: 'goldenrod'}
            for curr_band in bands:
                if bl_chrom_dic_for_cib[curr_band] is None: continue
                plot( bl_chrom_dic_for_cib[curr_band], color = color_dic[curr_band])
                plot( bl_chrom_dic_for_tsz[curr_band], color = color_dic[curr_band], ls = '-.')
            xlim(0., 10000.); ylim(0.8, 1.2)
            show(); sys.exit()


    if rs != -1: np.random.seed(rs)
    res_cib_a_arr, res_cib_b_arr = np.zeros( sa_arr.shape ), np.zeros( sa_arr.shape )
    for tmpsimno in range(total_sims_for_tsz_cib):

        """
        d1_tsz_cib_dic = sim_tsz_cib_estimate_dic[sim_or_data_tsz][m1][reqd_tsz_cib]
        d2_tsz_cib_dic = sim_tsz_cib_estimate_dic[sim_or_data_tsz][m2][reqd_tsz_cib]
        #d1_tsz_cib_dic = sim_tsz_cib_estimate_dic['sim_tsz'][m1][reqd_tsz_cib]
        #d2_tsz_cib_dic = sim_tsz_cib_estimate_dic['sim_tsz'][m2][reqd_tsz_cib]
        for tmpsimno in d1_tsz_cib_dic:
            if tmpsimno<25: continue
            curr_tsz_cib_est1 = 2*d1_tsz_cib_dic[tmpsimno]/1e6
            curr_tsz_cib_est2 = 2*d2_tsz_cib_dic[tmpsimno]/1e6
            sa_arr[tmpsimno][curr_rho_tsz_cib_linds] = sa_arr[tmpsimno][curr_rho_tsz_cib_linds] + curr_tsz_cib_est1[curr_rho_tsz_cib_linds]
            sb_arr[tmpsimno][curr_rho_tsz_cib_linds] = sb_arr[tmpsimno][curr_rho_tsz_cib_linds] + curr_tsz_cib_est2[curr_rho_tsz_cib_linds]
        """
        if cib_scatter_sigma is not None: #tweak CIB now
            curr_tweak_arr = 1. + np.random.standard_normal(len(bands)) * cib_scatter_sigma
            curr_tweak_arr[-2:] = 1.
        elif cib_cal_1 is not None:
            assert cib_cal_2 is not None
            assert cib_cal_3 is not None
            assert cib_cal_4 is not None
            assert cib_cal_5 is not None
            assert cib_cal_6 is not None
            curr_tweak_arr = [cib_cal_1, cib_cal_2, cib_cal_3, cib_cal_4, cib_cal_5, cib_cal_6]
        else:
            curr_tweak_arr = np.ones( len(bands ))

        #read maps and get spectra
        cl_tsz_cib_dic = {}
        cl_tsz_cib_dic['TT'] = {}
        for b1cntr, band1 in enumerate( bands ):
            for b2cntr, band2 in enumerate( bands ):

                ###if band1 != 150 or band2 != 150: continue

                binned_el = sim_ps_dic[tmpsimno][(band1, band1)]['binned_el']
                cl_cib_ori = sim_ps_dic[tmpsimno][(band1, band1)]['cib']
                cib_tweak_val_for_ps = np.sqrt( curr_tweak_arr[b1cntr] * curr_tweak_arr[b1cntr] )
                cl_cib = np.copy( cl_cib_ori ) * cib_tweak_val_for_ps
                cl_cib_tweak_only = np.copy( cl_cib_ori ) - np.copy( cl_cib )

                if sim_or_data_tsz == 'sim_tsz':
                    cl_tsz = sim_ps_dic[tmpsimno][(band2, band2)]['tsz']

                elif sim_or_data_tsz == 'cibmindata_tsz':
                    curr_tsz_compton_y_fac = sim_tsz_cib_estimate_dic['tsz_compton_y_fac'][band2] * sim_tsz_cib_estimate_dic['tsz_compton_y_fac'][band2]
                    cl_tsz = sim_tsz_cib_estimate_dic['cl_yy_fromcibmindata'] * curr_tsz_compton_y_fac * 1e6

                if include_beam_chromaticity:
                    curr_bl_chrom_cib, curr_bl_chrom_tsz = bl_chrom_dic_for_cib[band1], bl_chrom_dic_for_tsz[band2]
                    ##print('hi', band1, band2, curr_bl_chrom_cib, curr_bl_chrom_tsz)
                    if curr_bl_chrom_cib is not None:
                        dummy_els = np.arange( len(curr_bl_chrom_cib) )
                        curr_bl_chrom_cib = np.interp(binned_el, dummy_els, curr_bl_chrom_cib)
                    else:
                        curr_bl_chrom_cib = np.ones( len(binned_el) )
                    if curr_bl_chrom_tsz is not None:
                        dummy_els = np.arange( len(curr_bl_chrom_tsz) )
                        curr_bl_chrom_tsz = np.interp(binned_el, dummy_els, curr_bl_chrom_tsz)
                    else:
                        curr_bl_chrom_tsz = np.ones( len(binned_el) )
                    cl_cib = cl_cib / curr_bl_chrom_cib**2.
                    cl_tsz = cl_tsz / curr_bl_chrom_tsz**2.


                #tszXCIB
                if rho_tsz_cib is not None:
                    cl_cib_tsz = -rho_tsz_cib * np.sqrt( cl_cib * cl_tsz )
                else:
                    assert sim_or_data_tsz == 'sim_tsz'
                    cl_cib_tsz = sim_ps_dic[tmpsimno][(band1, band2)]['tsz_cib']

                ##print(band1, band2, curr_tsz_compton_y_fac, cl_cib, cl_tsz, cl_cib_tsz); ##sys.exit()

                cl_tsz_cib_dic['TT'][(band1, band2)] = cl_cib_tsz
            ##sys.exit()

        #CIB/tSZ
        cl_tsz_dic = {}
        cl_tsz_dic['TT'] = {}
        cl_cib_dic = {}
        cl_cib_dic['TT'] = {}
        cl_cib_tweaked_dic = {}
        cl_cib_tweaked_dic['TT'] = {}
        for b1cntr, band1 in enumerate( bands ):
            for b2cntr, band2 in enumerate( bands ):
                cl_cib_ori = sim_ps_dic[tmpsimno][(band1, band2)]['cib']
                cib_tweak_val_for_ps = np.sqrt( curr_tweak_arr[b1cntr] * curr_tweak_arr[b2cntr] )
                cl_cib = np.copy( cl_cib_ori ) * cib_tweak_val_for_ps
                cl_cib_tweak_only = np.copy( cl_cib_ori ) - np.copy( cl_cib )

                if sim_or_data_tsz == 'sim_tsz':
                    cl_tsz = sim_ps_dic[tmpsimno][(band1, band2)]['tsz']

                elif sim_or_data_tsz == 'cibmindata_tsz':
                    curr_tsz_compton_y_fac = sim_tsz_cib_estimate_dic['tsz_compton_y_fac'][band1] * sim_tsz_cib_estimate_dic['tsz_compton_y_fac'][band2]
                    cl_tsz = sim_tsz_cib_estimate_dic['cl_yy_fromcibmindata'] * curr_tsz_compton_y_fac * 1e6

                cl_tsz_dic['TT'][(band1, band2)] = cl_tsz
                cl_cib_dic['TT'][(band1, band2)] = np.copy( cl_cib_ori )
                cl_cib_tweaked_dic['TT'][(band1, band2)] = cl_cib_tweak_only
            ##sys.exit()

        if (0):
            ax = subplot(111, yscale = 'log')
            tmpels = array([ 750., 1250., 1750., 2250., 2750., 3250., 3750., 4250., 4750.])
            tmpdl_fac = tmpels * (tmpels+1)/2/np.pi * 1e6
            bbb1, bbb2 = 220, 857
            plot( tmpels, tmpdl_fac * cl_cib_tweaked_dic['TT'][(bbb1, bbb2)])
            ylim(0.1, 1e10)
            show()
            sys.exit()


        wl11, wl12 = wl_dic[m1[0]], wl_dic[m1[1]]
        wl21, wl22 = wl_dic[m2[0]], wl_dic[m2[1]]

        #residual tSZ x CIB
        curr_tsz_cib_est1 = get_ilc_residual_using_weights(cl_tsz_cib_dic, wl11, bands, wl2 = wl12, el = binned_el)
        curr_tsz_cib_est2 = get_ilc_residual_using_weights(cl_tsz_cib_dic, wl21, bands, wl2 = wl22, el = binned_el)
        curr_tsz_cib_est1 = 2*curr_tsz_cib_est1/1e6
        curr_tsz_cib_est2 = 2*curr_tsz_cib_est2/1e6
        if explicitly_null_tsz_cib_in_cibfree and m1 == ('ycibfree', 'ycibfree'):
            curr_tsz_cib_est1 = curr_tsz_cib_est1 * 0.
        if explicitly_null_tsz_cib_in_cibfree and m2 == ('ycibfree', 'ycibfree'):
            curr_tsz_cib_est2 = curr_tsz_cib_est2 * 0.

        #residual CIB (uncorrelated piece)
        curr_cib_est1 = get_ilc_residual_using_weights(cl_cib_dic, wl11, bands, wl2 = wl12, el = binned_el)
        curr_cib_est2 = get_ilc_residual_using_weights(cl_cib_dic, wl21, bands, wl2 = wl22, el = binned_el)
        curr_cib_est1 = curr_cib_est1/1e6
        curr_cib_est2 = curr_cib_est2/1e6
        res_cib_a_arr[tmpsimno] = curr_cib_est1
        res_cib_b_arr[tmpsimno] = curr_cib_est2
        """
        if uncorr_cib_frac_a is not None:
            assert uncorr_cib_frac_b is not None
            uncorr_cib_in_sa = uncorr_cib_frac_a * curr_cib_est1
            uncorr_cib_in_sb = uncorr_cib_frac_b * curr_cib_est2
        else:
            uncorr_cib_in_sa = np.zeros( len(curr_cib_est1) )
            uncorr_cib_in_sb = np.zeros( len(curr_cib_est2) )
        """

        #residual CIB due to scatter
        curr_cib_tweak_est1 = get_ilc_residual_using_weights(cl_cib_tweaked_dic, wl11, bands, wl2 = wl12, el = binned_el)
        curr_cib_tweak_est2 = get_ilc_residual_using_weights(cl_cib_tweaked_dic, wl21, bands, wl2 = wl22, el = binned_el)
        curr_cib_tweak_est1 = curr_cib_tweak_est1/1e6
        curr_cib_tweak_est2 = curr_cib_tweak_est2/1e6

        if (0):

            #tSZ for checking
            curr_tsz_est1 = get_ilc_residual_using_weights(cl_tsz_dic, wl11, bands, wl2 = wl12, el = binned_el)
            curr_tsz_est2 = get_ilc_residual_using_weights(cl_tsz_dic, wl21, bands, wl2 = wl22, el = binned_el)
            curr_tsz_est1 = curr_tsz_est1/1e6
            curr_tsz_est2 = curr_tsz_est2/1e6

            clf()
            tmpels = array([ 750., 1250., 1750., 2250., 2750., 3250., 3750., 4250., 4750.])
            tmpdl_fac = tmpels * (tmpels+1)/2/np.pi * 1e12
            plot(tmpels, tmpdl_fac * curr_tsz_est1)
            plot(tmpels, tmpdl_fac * curr_tsz_est2)
            plot(tmpels, tmpdl_fac * curr_cib_tweak_est1)
            plot(tmpels, tmpdl_fac * curr_cib_tweak_est2)
            xlim(0., 5000.); #ylim(-2., 2.)
            show(); sys.exit()


        if reqd_linds is None:
            reqd_linds = np.arange(len(curr_tsz_cib_est2))
        sa_arr[tmpsimno][reqd_linds] = sa_arr[tmpsimno][reqd_linds] + curr_tsz_cib_est1[reqd_linds] + curr_cib_tweak_est1[reqd_linds]# + uncorr_cib_in_sa[reqd_linds]
        sb_arr[tmpsimno][reqd_linds] = sb_arr[tmpsimno][reqd_linds] + curr_tsz_cib_est2[reqd_linds] + curr_cib_tweak_est2[reqd_linds]# + uncorr_cib_in_sb[reqd_linds]
        ###print(tmpsimno, np.mean(sa_arr[tmpsimno]), np.mean(sb_arr[tmpsimno]), np.mean(curr_tsz_cib_est1), np.mean(curr_tsz_cib_est2), curr_tweak_arr); sys.exit()
        if res_tsz_cib_a_arr is not None:
            res_tsz_cib_a_arr[tmpsimno][reqd_linds] = np.copy(curr_tsz_cib_est1[reqd_linds])/2
            res_tsz_cib_b_arr[tmpsimno][reqd_linds] = np.copy(curr_tsz_cib_est2[reqd_linds])/2

    if res_tsz_cib_a_arr is not None and res_tsz_cib_b_arr is not None:
        return sa_arr, sb_arr, res_cib_a_arr, res_cib_b_arr, res_tsz_cib_a_arr, res_tsz_cib_b_arr
    else:
        return sa_arr, sb_arr, res_cib_a_arr, res_cib_b_arr

def inject_res_tsz_cib(els, ilckeyname, cl_yy, cib_sys_key = 'cib_tweaked_spt_only_max_tweak_0.2', rho_tsz_cib = -0.2):
    cl_cib_res_arr = res_dic[ilckeyname]['cl_sys_dic'][cib_sys_key]
    cl_cib_res_mean = np.mean( cl_cib_res_arr, axis = 0)
    ##print(cib_res_mean)
    
    cl_cib_y = rho_tsz_cib * np.sqrt(cl_yy * cl_cib_res_mean) 
    return cl_cib_res_mean, cl_cib_y
    
def get_null_test_chi_sq(data, cov):
    cov_inv = np.linalg.inv(cov)
    chi_sq_val = np.dot(data, np.dot(data, cov_inv))
    if (0):
        data_err = np.sqrt( np.diag( cov) )
        chi_sq_val = np.sum(data**2. / data_err**2.)

    chi_val = np.dot(data, cov_inv)
    data_err = np.sqrt( np.diag( cov) )
    chi_val = data / data_err**2.
    return chi_sq_val, chi_val

def get_sim_arrary(res_dic, ilc_keyname, which_sim):
    if which_sim == 'cl_arr_for_non_gau_cov':
        sim_arr = np.asarray( res_dic[ilc_keyname][which_sim] )
    elif which_sim == 'cib_cmb_rad_tsz_ksz_noise_rc5.1_noslope_spt3gbeams_compdependent_full':
        sim_arr = np.asarray( res_dic[ilc_keyname]['for_consistency_tests'][which_sim][0] )
    elif which_sim == 'cib_cmb_rad_tsz_ksz_noise_rc5.1_noslope_spt3gbeams_compdependent_half_splits':
        sim_arr = np.asarray( res_dic[ilc_keyname]['for_consistency_tests'][which_sim][0] )
    elif which_sim == 'cmb_tsz_ksz_noise_uncorrcib_uncorrrad_rc5.1_noslope_spt3gbeams_compdependent_full':
        sim_arr = np.asarray( res_dic[ilc_keyname]['for_consistency_tests'][which_sim][0] )
    elif which_sim == 'cmb_tsz_ksz_noise_uncorrcib_uncorrrad_rc5.1_noslope_spt3gbeams_compdependent_half_splits':
        sim_arr = np.asarray( res_dic[ilc_keyname]['for_consistency_tests'][which_sim][0] )
    else:
        print(res_dic.keys(), ilc_keyname); sys.exit()
        sim_arr = np.asarray( res_dic[ilc_keyname][which_sim][0] )        
    return sim_arr

def corr_from_cov(covmat):
    diags = np.sqrt(np.diag(covmat))
    corrmat = np.zeros_like(covmat)
    for i in range(covmat.shape[0]):
        for j in range(covmat.shape[0]):
            corrmat[i, j] = covmat[i, j] / (diags[i] *  diags[j])
    return corrmat

def get_inverse_covariance(bands, elcnt, cl_dict, return_cov=False):
    """
    Get the inverse band-band covariance matrix at each el

    Parameters
    ----------
    bands : array
        array of frequency bands for which we need the covariance.
    elcnt : int
        ell index.
    cl_dict : dict
        dictionary containing (signal+noise) auto- and cross- spectra of different freq. channels.
    return_cov : bool
        if True, this functions returns cov along with its inverse.
        default is False.

    Returns
    -------
    cov_inv: array
        inverse of the band-band covariance matrix at each ell. dimension is nband x nband.
    """
    cov = np.mat(create_covariance(bands, elcnt, cl_dict))
    if np.sum(cov) == 0.0:
        return None
    cov_inv = np.linalg.pinv(cov)

    if return_cov:
        return cov_inv, cov
    else:
        return cov_inv


def create_covariance(bands, elcnt, cl_dict):

    """
    Creates band-band covariance matrix at each el

    Parameters
    ----------
    bands : array
        array of frequency bands for which we need the covariance.
    elcnt : int
        ell index.
    cl_dict : dict
        dictionary containing (signal+noise) auto- and cross- spectra of different freq. channels.

    Returns
    -------
    cov: array
        band-band covariance matrix at each ell. dimension is nband x nband.
    """

    nc = len(bands)
    nspecs, specs = get_teb_spec_combination(cl_dict)
    cov = np.zeros((nspecs * nc, nspecs * nc))

    for specind, spec in enumerate(specs):
        curr_cl_dict = cl_dict[spec]
        if nspecs == 1:  # cov for TT or EE or BB
            for ncnt1, band1 in enumerate(bands):
                for ncnt2, band2 in enumerate(bands):
                    j, i = ncnt2, ncnt1
                    cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
        else:  # joint or separate TT/EE constraints #fix me: include BB for joint constraints.
            if spec == 'TT':
                for ncnt1, band1 in enumerate(bands):
                    for ncnt2, band2 in enumerate(bands):
                        j, i = ncnt2, ncnt1
                        cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
            elif spec == 'EE':
                for ncnt1, band1 in enumerate(bands):
                    for ncnt2, band2 in enumerate(bands):
                        j, i = ncnt2 + nc, ncnt1 + nc
                        cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
            elif spec == 'TE':
                for ncnt1, band1 in enumerate(bands):
                    for ncnt2, band2 in enumerate(bands):
                        j, i = ncnt2 + nc, ncnt1
                        cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
                        cov[i, j] = curr_cl_dict[(band1, band2)][elcnt]

    return cov


def get_teb_spec_combination(cl_dict):

    """
    uses cl_dict to determine if we are using ILC jointly for T/E/B.

    Parameters
    ----------
    cl_dict : dict
        dictionary containing (signal+noise) auto- and cross- spectra of different freq. channels.

    Returns
    -------
    nspecs : int
        tells if we are performing ILC for T alone or T/E/B together.
        default is 1. For only one map component.

    specs : list
        creates ['TT', 'EE', 'TE', ... etc.] based on cl_dict that is supplied.
        For example:
        ['TT'] = ILC for T-only
        ['EE'] = ILC for E-only
        ['TT', 'EE'] = ILC for T and E separately.
        ['TT', 'EE', 'TE'] = ILC for T and E jointly.
    """

    # fix-me. Do this in a better way.
    specs = sorted(list(cl_dict.keys()))

    if specs == ['TT'] or specs == ['EE'] or specs == ['BB']:  # only TT is supplied
        nspecs = 1
    elif specs == sorted(['TT', 'EE']) or specs == sorted(
        ['TT', 'EE', 'TE']
    ):  # TT/EE/TE are supplied
        nspecs = 2
    elif specs == sorted(['TT', 'EE', 'BB']) or specs == sorted(
        ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    ):  # TT/EE/BB are supplied
        nspecs = 3
    else:
        logline = 'cl_dict must contain TT/EE/BB spectra or some combination of that'
        raise ValueError(logline)

    return nspecs, specs

def get_ilc_residual_using_weights(cl_dic, wl, bands, wl2 = None, lmax = 10000, el = None):
    if wl2 is None:
        wl2 = wl
    if el is None:
        el = np.arange(lmax)
    res_ilc = []
    for elcnt, currel in enumerate(el):
        clmat = np.mat( create_covariance(bands, elcnt, cl_dic) )
        currw_ilc1 = np.mat( wl[:, elcnt] )
        currw_ilc2 = np.mat( wl2[:, elcnt] )
        curr_res_ilc = np.asarray(np.dot(currw_ilc1, np.dot(clmat, currw_ilc2.T)))[0][0]
        res_ilc.append( curr_res_ilc )

    res_ilc = np.asarray(res_ilc)
    res_ilc[np.isnan(res_ilc)] = 0.
    res_ilc[np.isinf(res_ilc)] = 0.

    return res_ilc

def get_likelihood_simple(data, model, error):
    logLval = 0.
    for (d, m, e) in zip(data, model, error):
        curr_logLval = -0.5 * (d-m)**2. / e**2.
        logLval += curr_logLval
    return logLval

def get_likelihood_from_loglikelihood(logL):
    logL = np.asarray( logL )
    logL = logL - np.max(logL)
    L = np.exp( logL )
    L /= np.max(L)
    return L

def get_likelihood(data, model, cov):
    """
    function to calculate the likelihood given data, model, covariance matrix
    """
    data = np.asarray( data )
    model = np.asarray( model )
    cinv = np.linalg.inv(cov)
    residual = data - model
    logLval =  -0.5 * np.asarray( np.dot(residual.T, np.dot( cinv, residual ))).squeeze()
    return logLval
    
def get_binning_operators(ells, ell_bins, ell_weights = None, min_ell = 2, use_dl_for_binning_operators = True, epsilon_for_diag = 1e-8):
    
    #Eqs. (20) and (21) of https://arxiv.org/pdf/astro-ph/0105302.pdf
    total_ells = len(ells)

    if ell_weights is None: ell_weights = np.ones_like(ells)
    total_ell_bins = len( ell_bins )
    pbl = np.zeros( (total_ell_bins, total_ells) ) #N_binned_el x N_ells (Eq. 20 of https://arxiv.org/pdf/astro-ph/0105302.pdf)
    qlb = np.zeros( (total_ells, total_ell_bins) ) #N_ells x N_binned_el (Eq. 21 of https://arxiv.org/pdf/astro-ph/0105302.pdf)
    
    '''    
    epsilon_diag_mat_for_pbl = np.eye( total_ell_bins, total_ells  ) * epsilon_for_diag
    epsilon_diag_mat_for_qlb = np.eye( total_ells, total_ell_bins  ) * epsilon_for_diag

    pbl = pbl + epsilon_diag_mat_for_pbl
    qlb = qlb + epsilon_diag_mat_for_qlb
    '''    

    for bcntr, (b1, b2) in enumerate(ell_bins):
        ##linds = np.where( (ells>=b1) & (ells<=b2) )[0]
        linds = np.where( (ells>=b1) & (ells<=b2) )[0]
        b3 = b2+1
        ##print(ells[linds], b1, b2); sys.exit()
        if len(linds) == 0 or b2<min_ell: continue #make sure \ell >= min_ell.
        if use_dl_for_binning_operators:
            dl_fac = ell_weights[linds] * ells[linds] * (ells[linds]+1)/2/np.pi
        else:
            dl_fac = 1. * ell_weights[linds]

        #pbl[bcntr, linds] = dl_fac * ell_weights[linds] / np.sum(ell_weights[linds]) ##(b2-b1)
        ##print(dl_fac, linds); sys.exit()
        pbl[bcntr, linds] = dl_fac / (b3-b1)
        qlb[linds, bcntr] = 1./dl_fac
        ##print(bcntr, b1, b2, b3, len(linds), pbl[bcntr, linds]/dl_fac)#, ells[linds], dl_fac)#, pbl[bcntr])
        ##print(bcntr, b1, b2, ells[linds]); ##sys.exit()

    return pbl, qlb

def get_bpwf(pbl, qlb, mll, bl = None, fl = None):
    kll = mll
    if bl is not None:
        kll = np.dot( kll, bl**2.)
    if fl is not None:
        kll = np.dot( kll, fl)
    ##print(qlb); sys.exit()
    kbb = np.dot(pbl, np.dot(kll, qlb))
    ##print(kbb[1]); ##sys.exit()
    kbb_inv = np.linalg.inv(kbb) #inverse of kbb
    #print(kbb_inv); 
    ##sys.exit()
    bpwf = np.dot(kbb_inv, np.dot(pbl, kll) ) #Eq. (25) of https://arxiv.org/pdf/1707.09353.

    return bpwf, kbb, kbb_inv

def get_ell_bin(el_unbinned, delta_el, lmin = None, lmax = None):
    ##if lmin is None: lmin = min(el_unbinned)
    ##if lmax is None: lmax = max(el_unbinned)
    lmin, lmax = 1, max(el_unbinned)
    el_binned = np.arange(lmin, lmax+delta_el, delta_el)
    ell_bins = [(b-delta_el/2, b+delta_el/2) for b in el_binned]
    ##ell_bins = [(b, b+delta_el) for b in el_binned]
    ##print(ell_bins); sys.exit()
    
    return el_binned, ell_bins

def perform_binning(el_unbinned, cl_unbinned, delta_el = 100, return_dl = False, bl = None, fl = None, lmin = 30, lmax = 5000, epsilon_for_diag = 1e-8, debug = False):
    #ell_bins = [(b, b+delta_el) for b in el_binned]
    el_binned, ell_bins = get_ell_bin(el_unbinned, delta_el, lmin = lmin, lmax = lmax)
    ##print(el_binned.shape, len(ell_bins)); sys.exit()

    reclen_binned = len( el_binned )
    reclen_unbinned = len( el_unbinned )
    ell_weights = np.ones( reclen_unbinned )
    #ell_weights[el_unbinned<lmin] = epsilon_for_diag
    #ell_weights[el_unbinned>lmax] = epsilon_for_diag
    pbl, qlb = get_binning_operators(el_unbinned, ell_bins, ell_weights = ell_weights, use_dl_for_binning_operators = return_dl)
    mll = np.diag( np.ones( reclen_unbinned ) )

    #lmin/lmax cuts
    unbinned_inds_to_cut = np.where( (el_unbinned<lmin) & (el_unbinned>lmax) )
    binned_inds_to_cut = np.where( (el_binned<lmin) & (el_binned>lmax))
    cl_unbinned[(el_unbinned<lmin) | (el_unbinned>lmax)] = 0. #lmin/lmax cut
    
    epsilon_diag_mat = np.eye( reclen_unbinned )# * epsilon_for_diag
    mll[(el_unbinned<lmin) | (el_unbinned>lmax)] = 0. #adding a lmin/lmax cut.
    mll = mll + epsilon_diag_mat
    #pbl[[el_binned<lmin, None], el_unbinned<lmin] = 0.
    #qlb[el_unbinned<lmin] = 0.

    ##from IPython import embed; embed()
    bpwf, kbb, kbb_inv = get_bpwf(pbl, qlb, mll, bl = bl, fl = fl)

    pspec_binned = np.dot(kbb_inv, np.dot(pbl, cl_unbinned) ) #Eq. (26) of https://arxiv.org/pdf/astro-ph/0105302.pdf. Note that there is no noise bias here.
    ##pspec_binned[(el_binned<lmin) | (el_binned>lmax)] = 0. #lmin/lmax cut
    ##print( el_binned, lmin, lmax, pspec_binned ); ##sys.exit()

    ###from IPython import embed; embed()

    if debug:
        ax = subplot(111, yscale = 'log')
        plot( el_unbinned, cl_unbinned, color = 'black' )
        plot( el_binned, pspec_binned, color = 'orangered' )
        show()
        color_arr = [cm.jet(int(d)) for d in np.linspace(0., 255, len(bpwf))]

        for b in range(len(bpwf)):
            plot( el_unbinned, bpwf[b], color = color_arr[b])
        show()

    ##from IPython import embed; embed();
    
    return el_binned, pspec_binned, bpwf


def perform_binning_simple(el, cl, lmin = None, lmax = None, delta_el = 50): 
    if lmin is None: lmin = min(el)
    if lmax is None: lmax = min(el)
    binned_el = np.arange(lmin, lmax+delta_el, delta_el)
    binned_cl = np.zeros( len(binned_el) )   
    for lll, l1 in enumerate( binned_el ):
        l2 = l1 + delta_el
        linds = np.where( (el>=l1) & (el<l2) )[0]
        binned_cl[lll] = np.mean( cl[linds])
    return binned_el, binned_cl
