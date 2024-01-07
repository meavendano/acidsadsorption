# Script to determine lumped apparent mass transfer coefficients (kapps) based on empirical and semi-empirical correlations
# Originally created and developed by Marco Avendano in Oct 2022
# Bibliographic information:
# (1) Schmidt-Traub, H.; Schulte, M.; Seidel-Morgenstern, A. Modeling of Chromatographic Processes. In Preparative Chromatography, 3rd ed.; John Wiley  Sons, 1993; pp 311−350.
# (2) Taylor, R.; Krishna, R. Estimation of Diffusion Coefficients. In Multicomponent Mass Transfer, 3rd ed.; Wiley-VCH, 2020; pp 67−93.

#Define bed parameters refer to manuscript for more information
ep=.66 #Particle porosity (cm3 particle void volume / cm3 of bulk particle volume)
eb=.44 #Bed porosity (cm3 bed void volume/cm3 bed volume)
v=10 #(cm/min) Approximate SMB intersitial velocity (v=Q/A*eb)
dp=1190e-4 #(cm) mean particle diameter/size (~10 mesh = 1190 microns)

def Dm(T,rho,M_solute,solvent):
    #Function that return the diffusivity coefficient, Dm (cm2/min), of a
    #solute on a given solvent, based on the Wilke-Chang correlation.
    
    if solvent=='Water':
        #Assume Water as the Solvent
        M=18 #Molar Weight g/mol
        mu=.95 #Solvent viscosity cP
        a=2.26 #Association Factor
    elif solvent=='Methanol':
        #Assume Methanol as the Solvent
        M=32 #Molar Weight g/mol
        mu=.55 #Solvent viscosity cP
        a=1.9 #Association Factor
    
    V=M_solute/rho #Molar volume of the solute in cm3/mol
    return 7.4e-8*(a*M)**.5*T/(mu*V**.6)*60

def kpore(Dm,ep,dp):
    #Function that returns the internal mass transfer resistance, kpore (min^-1)
    #of a given species.
    Dpore=ep/(2-ep)**2*Dm #pore diffusivity in cm2/min
    return 10*ep*Dpore/dp

def kfilm(Dm,ep,eb,v,dp):
    #Function that returns the external mass transfer resistance, kfilm (min^-1)
    #of a given species.
    return 1.09/eb*Dm/dp*(eb*v*dp/Dm)**.33
    
def keff(kpore,kfilm):
    #Function that returns the effective mass transfer resistance, keff (min^-1)
    #of a given species.
    return (1/kpore+1/kfilm)**-1
    
def kapp(keff):
    #Function that returns the apparent mass transfer resistance, kapp (min^-1)
    #of a given species.
    return keff*3/(dp/2)

T=30+273 #Temperature K

solutes=['Glycolic Acid','Malic Acid']
solvents=['Water','Methanol']

for solvent_i in solvents:
    print(f'\n{solvent_i} as the Solvent:')
    for solute_i in solutes:
        
        if solute_i == 'Glycolic Acid':
            rho_i=1.5 #solute density cm3/g
            M_i=76 #solute molecular weight g/mol
        elif solute_i == 'Malic Acid':
            rho_i=1.6 #solute density cm3/g
            M_i=140 #solute molecular weight g/mol
        
        Dm_i=Dm(T,rho_i,M_i,solvent_i)
        kfilm_i=kfilm(Dm_i,ep,eb,v,dp)
        kpore_i=kpore(Dm_i,ep,dp)
        keff_i=keff(kpore_i,kfilm_i)
        kapp_i=kapp(keff_i)
        
        print(f'{solute_i}\nDm = {Dm_i/60:.2e} cm2/s, kfilm = {kfilm_i:.2f} min^-1, kpore = {kpore_i:.2f} min^-1, keff = {keff_i:.2f} min^-1, kapp = {kapp_i:.2f} min^-1')