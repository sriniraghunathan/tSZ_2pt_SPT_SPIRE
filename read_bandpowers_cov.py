import numpy as np

#numpy file
fname = 'publish/bandpowers_and_cov/spt_100d_dl_yy_bandpowers_and_cov_all_estimators.npy'
spt_100d_yy_dict = np.load(fname, allow_pickle = True).item()
print( '\n100d SPT yy results.\n\nEstimators are: ', list(spt_100d_yy_dict.keys()) )

for m1m2 in spt_100d_yy_dict:
    m1, m2 = m1m2
    print('\n\tEstimator = %sx%s' %(m1, m2))
    el_eff = spt_100d_yy_dict[m1m2]['el_eff']
    dl_yy = spt_100d_yy_dict[m1m2]['bandpower'] #Dl_yy [1e12]
    dl_yy_err = spt_100d_yy_dict[m1m2]['bandpower_error'] #Dl_yy_err [1e12] 
    dl_yy_cov = spt_100d_yy_dict[m1m2]['cov'] #Dl_yy_cov [1e12]
    print('\t\t', 'el_eff=', el_eff, 'dl_yy=', dl_yy, 'dl_yy_err=', dl_yy_err)

print('\nDone.\n')

