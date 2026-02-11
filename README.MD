# tSZ 2pt analysis using SPT x SPIRE 

## Paper:
* Raghunathan S., SPT-3G and SPTpol Collaboration 2026; arXiv:[2602.10107](https://arxiv.org/abs/2602.10107). 

## Overview:
* In this work, we reconstruct the full shape of the thermal Sunyaev–Zeldovich (tSZ) or Compton-$y$ power spectrum by optimally combining data from the South Pole Telescope (SPT) and *Herschel*-SPIRE surveys. 
* We also reconstruct the scale-dependent correlation between tSZ and CIB signals.
* We release the bandpowers, maps, and other associated data products here.

## Results:

### Code:
* [read\_maps\_masks\_bandpowers\_cov.ipynb](https://github.com/sriniraghunathan/tSZ_2pt_SPT_SPIRE/blob/main/read_maps_masks_bandpowers_cov.ipynb): Please look into this notebook to read the data products.
	* Html rendering of the above notebook: [Link](https://pole.uchicago.edu/public/data/tsz_2pt_raghunathan26/read_maps_masks_bandpowers_cov.html)
* Please also check this [SPT_data_link](https://pole.uchicago.edu/public/data/tsz_2pt_raghunathan26/) for additional details about the data products.

### Maps:
<p align="left">
  <img src="plots/ymap_plots_with_zoom.png" width="500" title="Compton-$y$" map>
</p>

* We release two versions of the Compton-$y$ maps. Both these versions have the "full coadd", as well as two half-splits. These can be found inside [maps_and_masks](https://github.com/sriniraghunathan/tSZ_2pt_SPT_SPIRE/tree/main/maps_and_masks) folder.
	* MV: minimum-variance (MV) linear combination (LC).
	* CIB-min: The LC combination in which the cosmic infrared background (CIB) signal is minimised.

### Bandpowers:
<p align="left">
	  <img src="plots/ymap_spectra_data_plus_others.png" width="500" title="Compton-$y$ power spectrum">
</p>

* In [bandpowers_and_cov](https://github.com/sriniraghunathan/tSZ_2pt_SPT_SPIRE/tree/main/bandpowers_and_cov) folder, you can find the associated bandpowers along with the bandpower covariance matrix and the bandpower window functions. The folder contains:
	* Auto-power spectrum of:
		* CIB-min
		* MV
	* Cross-spectrum of CIB-min and MV maps.

### tSZ $\times$ CIB cross-correlation:
<p align="left">
  <img src="plots/rho_tsz_cib_corner_6bins.png" width="300" title="tSZ x CIB correlation corner plot">
  <img src="plots/rho_tsz_cib.png" width="400" title="tSZ x CIB correlation">
</p>

* We reconstruct the scale-dependent tSZ $\times$ CIB cross-correlation. These can also be found in the [bandpowers_and_cov](https://github.com/sriniraghunathan/tSZ_2pt_SPT_SPIRE/tree/main/bandpowers_and_cov) folder.
