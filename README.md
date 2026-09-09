# ESM3ΔG monomer improves discrimination of functional, de novo designed protein minibinders

This repository contains all of the necessary scripts to score structures by ESM3dG and run retrospective success rate and binding/non-binding binning analysis.

These scripts require setting up ESM3dG as per Cho, et al (https://github.com/yehlincho/absolute-stability-predictor)

Input structures may be in .cif or .pdb format

Analysis scripts (Combined_violin_by_length_monomer_wTtests_colorbysource.py, SingleSource_violin_by_length_monomer_wTtests.py, and retrospective_success_ParetooptdG.py) are set up to be compatible with dG outputs of ESM3dG run in monomer mode. They can be edited to be used with complex mode by replacing all instances of dG_ensemble with dG_AB_ensemble or dG_binder_ensemble.