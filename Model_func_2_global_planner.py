### this file solves the equilibrium prices, allocations - for given wages, policy values, wedges
### ANY number of countries and sectors
### Last updated: Feb 26, 2025

import matplotlib.pyplot as plt
import matplotlib.axes as axs
import numpy as np
import pandas as pd
import networkx as nx 
import textwrap
import seaborn as sns
import random
from scipy.optimize import fsolve





def solver(W_init,params,L,alp,gam,chi_mat,tau_mat,tau_L,solve_eq):
    C=params[0] #number of countries
    J=params[1] #number of industries
    epsilon=params[2] #production side elasticity of substitution
    sigma=params[3]


    ## Step 1: solve price system of equations
    W=np.concatenate((np.array([1]),W_init), axis=0).reshape(-1,1)

    
    #define coefficient matrix
    M_P=(alp*gam)*(np.ones((C*J, C*J))-tau_mat+chi_mat)**(1-epsilon)
    a=np.eye(C*J)-M_P

    # Compute spectral radius
    eigenvalues = np.linalg.eigvals(M_P)
    spectral_radius = max(abs(eigenvalues))
    #print("Spectral radius of price system matrix: " + str(spectral_radius))

    #define Wage vector
    W_CJ=np.repeat(W, J).reshape(-1,1)  #repeat wage for each industry within each country
    W_P=(1-alp)*((np.ones((C*J,1))-tau_L)*W_CJ)**(1-epsilon)  

    #solve linear system for vector of prices
    P_exp=np.linalg.inv(a)@W_P ## this returns vector of P^(1-epsilon)
    #P_temp=np.linalg.solve(a, np.transpose(W_P)) ## equivalent
    #numpy bug/feature in evaluating exponent with fraction sometimes returns nan even if answer is not complex. Explained here:  https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
    P_vec=np.sign(P_exp) * (np.abs(P_exp)) ** (1/(1-epsilon))


    #step 2: compute shares
    #compute consumer price index
    P_h=np.sum(P_vec**(1-sigma))**(1/(1-sigma))

    #compute HH expenditure shares
    s_h=(P_vec/P_h)**(1-sigma)   #add up to 1
    s_h_mat = np.tile(np.transpose(s_h), (C, 1))  # Repeat vector C times as rows
    ##dimensions for this are row country C buys from country-sector columns 
    ##because all countries face the same prices for each country-sector (ie. no trade-costs/policies/wedges in HH problem) each country consumes the same bundle (scaled by wage)


    #compute labour cost share
    s_L= W_P/P_exp
    #compute cost shares for intermediates that i buys from j
    s_ij=np.zeros((C*J,C*J))
    for ic in range(0,C*J):
        s_ij[ic,:]=((M_P[ic,:].reshape(-1,1)*P_exp)/P_exp[ic]).reshape(1,-1)    #not the best way of doing this but it works....

    #np.sum(s_ij,axis=1).reshape(-1,1)+s_L #check that sum=1 
    #np.sum(s_h_mat,axis=1)   #check that sum=1 

    """   
    #step 3: Solve demand system ----- n+1 retry 
    """
    #define coefficient matrix, in pieces first
    #feb17
    denom=np.ones((C*J, C*J))-tau_mat+chi_mat

    #a piece: the coefficient on R_ijcc', as function of R_ic
    a_1=np.transpose(s_ij)/np.transpose(denom)

    #b piece: related to the DWL (Pi_c)
    b_1 = chi_mat*s_ij/denom
    b_2 = np.repeat(np.transpose(np.sum(b_1,axis=1).reshape(-1,1)),C*J,axis=0)
    

    #c piece: related to the labour subsidies
    c_1 = -tau_L*s_L/(1-tau_L)
    c_2 = np.repeat(np.transpose(c_1),C*J,axis=0)

    #d piece: related to the subsidies on intermediates
    d_1 = -tau_mat*s_ij/denom
    d_2 = np.repeat(np.transpose(np.sum(d_1,axis=1).reshape(-1,1)),C*J,axis=0)

    #create hh consumption shares into a square matrix
    s_h_sq=np.transpose(np.repeat(s_h_mat,J,axis=0))

    ##Define coefficient matrix
    M_R =a_1+(b_2+c_2+d_2)*s_h_sq
    b=np.eye(C*J)-M_R
    eigenvalues = np.linalg.eigvals(M_R)
    spectral_radius = max(abs(eigenvalues))

    ## Define wage vector
    WL=W*L.reshape(-1,1)  #total income for each country
    W_R = np.transpose(s_h_mat)@WL  


    R_ic=np.linalg.inv(b)@W_R  ## this returns vector of R^c_i
    
    """
    ##step 4: Construct matrices for R_ijcc' (R_mat), Y_c, Pi_c 
    """
    R_mat=s_ij/denom*R_ic
    Pi_c=np.sum(np.reshape(np.sum(chi_mat*R_mat,axis=1),(C,J)),axis=1)

    L_subsidy_cost=np.sum(np.reshape((tau_L/(1-tau_L))*s_L*R_ic,(C,J)),axis=1) #labour subsidy cost
    M_subsidy_cost=np.sum(np.reshape(np.sum(tau_mat*R_mat,axis=1),(C,J)),axis=1) #intermediates subsidy cost
    wage_bill=W*L.reshape(-1,1)
    Y_c=wage_bill-L_subsidy_cost.reshape(-1,1)-M_subsidy_cost.reshape(-1,1)
    """
    ##Step 5: Compute the TB for each country
    """
    #imports country 2
    #imports=np.sum((Y_c[1]+Pi_c[1])*s_h_mat[1,:])+np.sum(R_mat[J:,0:J])
    #exports country 2
    #exports=np.sum((Y_c[0]+Pi_c[0])*s_h_mat[0,:])+np.sum(R_mat[0:J,J:])
    #TB=exports-imports
    

    ##create arrays with length number of country-sectors and number of countries
    arr_cj = np.arange(0, C*J) 
    arr_c = np.arange(0, C)
    #create empty placeholders for import, export vecs
    exports=np.zeros(C)
    imports=np.zeros(C)

    for cc in range(1,C+1):

        ##cc is the domestic country (cc-1, because indexing starts at 0)
        foreign_sectors = np.concatenate((arr_cj[:(cc-1)*J], arr_cj[cc*J:]))   # Select all elements except domestic sectors (cc-1)*J:cc*J
        domestic_sectors = arr_cj[(cc-1)*J:cc*J]   #select all sectors in cc
        countries_except_cc= np.concatenate((arr_c[:cc-1], arr_c[cc:]))  # Select all countries except cc

        #exports 
        exports_intermeds=np.sum(R_mat[np.ix_(foreign_sectors,domestic_sectors)])    #EXPORT=>  [ foreign buyer, domestic seller ]
        exports_cons=np.sum((Y_c[countries_except_cc]+Pi_c[countries_except_cc].reshape(-1,1))*s_h_mat[np.ix_(countries_except_cc, domestic_sectors)])  ###CHANGE HERE:  MAR6 --VG 
        exports[cc-1]=exports_cons+exports_intermeds

        #imports
        imports_intermeds=np.sum(R_mat[np.ix_(domestic_sectors,foreign_sectors)])  #IMPORT=>  [ dom buyer, foreign seller ]
        imports_cons=np.sum((Y_c[cc-1]+Pi_c[cc-1])*s_h_mat[cc-1, foreign_sectors])

        imports[cc-1]=imports_cons+imports_intermeds


    #trade balance for n-1 countries  
    TB=exports[1:]-imports[1:]
    TB_c0=exports[0]-imports[0] #check that the first country also has TB=0

    
    
    
    """
    ##step 6: Compute consumption in each country
    """
    #C_c=(Y_c-Pi_c.reshape(-1,1))/P_h
    
    if solve_eq==1:
        return TB
    else:
        return Y_c, Pi_c, P_h, P_vec, R_mat, R_ic, s_ij, s_L, s_h_mat, TB, TB_c0
    