# GAS
Generalized association study for matched non-Gaussian data
The GAS_matlab folder contains main MATLAB functions for implementing the proposed methods under the Generalized Association Study (GAS) framework. It provides a tool box for flexibly analyzing the association between two data sets with heterogeneous data types. 

Sim_Setting.m 			The m file contains all the simulation settings used in the paper.

Sim_RankEst.m            	The m file provides a demo showing how to use the main functions to estimate the latent ranks of heterogeneous matrices. 

Sim_ParamEst_dense.m		The m file provides a demo showing how to use the proposed methods to estimate model parameters.





Main functions in GAS_matlab:

GAS.m				The function estimates the joint and individual structure in the GAS model. It contains several variants such as the one-step approximation, and the sparse estimation. 

GAS_PermTest.m			The function conducts permutation tests to evaluate the statistical significance of the association coefficient.

ExpPCA.m			The function implements the exponential family PCA method. 

Nfold_CV_Mixed.m 		The function conducts N-fold cross validation to select the best latent rank for dual-typed data.

Nfold_CV_Single.m               The function conducts N-fold cross validation to select the best latent rank for a single data matrix.




Contact: Gen Li, PhD
         Assistant Professor of Biostatistics, Columbia University
         Email: gl2521@columbia.edu  
CopyRight all reserved
Last updated: 2/11/2017
