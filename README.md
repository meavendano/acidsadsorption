=====================================================================
Repository: Modeling for Recovery and Enrichment of Organic Acids from Kraft Black Liquor by Simulated Moving Bed Adsorption
=====================================================================

Journal: ACS Sustainable Chemistry & Engineering
Title: "Recovery and Enrichment of Organic Acids from Kraft Black Liquor by Simulated Moving Bed Adsorption"
Author(s): Fu, Qiang; Avendano, Marco; Bentley, Jason; Tran, Quyen; Shofner, Meisha; Sinquefield, Scott; Realff, Matthew; Nair, Sankar

Overview:
---------
This repository contains the code used for modeling in the manuscript titled "Recovery and Enrichment of Organic Acids from Kraft
Black Liquor by Simulated Moving Bed Adsorption" published in ACS Sustainable Chemistry & Engineering. The modeling approach
involves setting the adsorption as a system of partial differential equations in Pyomo — Optimization Modeling in Python.The result is
a framework expressed as a non-linear problem (NLP) solved using Ipopt 13.12.

Installation Instructions:
--------------------------
The installation of Pyomo and Ipopt can be done following the guidelines provided in Suzuki and Kawajiri 2021. Refer to the respective documentation for installation details.

Files Included:
---------------
1. SingleBedAdsorptionFitting_TracerWater.py:
   - Description: Fitting of the Single Bed Adsorption Breakthrough with water as a tracer. Estimates Pe and the water
	mass transfer coefficient.

2. SingleBedAdsorptionFitting_MethanolWater.py:
   - Description: Fitting of the Single Bed Adsorption Breakthrough for a methanol/water feed. Estimates kapp of methanol.

3. SingleBedAdsorptionFitting_GA_MA.py:
   - Description: Fitting of the Single Bed Adsorption Breakthrough for a Glycolic Acid/Malic Acid/water feed. Estimates kapp
	of Glycolic Acid (GA) and Malic Acid (MA).

4. SMB_Acids_Initialization_Fitting_Optimization.py:
   - Description: Initialization, Fitting, and Optimization subroutines for multiple SMB runs with feed containing
	Glycolic Acid/Malic Acid/methanol/water. Estimates kapp of all components based on the initial guess.

5. kapp_correlation.py:
   - Description: Based on correlations, estimates an initial guess of the kapp for Glycolic Acid (GA) and Malic Acid (MA).

Usage Instructions:
--------------------
Each Python file represents specific modeling scenarios. Refer to individual file comments or documentation for usage instructions,
required data input, configurations, and execution steps.

Contact Information:
--------------------
For inquiries or assistance, please contact:
- Avendano, Marco: mavendano6@gatech.edu

Please cite our manuscript if you find this code useful in your research:

Acknowledgment:
---------------
As acknowledged in the the files, the Pyomo modeling was based on the work from Suzuki and Kawajiri in 2021
