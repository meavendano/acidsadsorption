# Developed and original work by Marco Avendano in June 2022
# Code for the Adsorption Breakthrough of Water and Methanol

from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.util.model_size import build_model_size_report

# -------------------------------------------------------------------
####Define Parameters and Constants
# -------------------------------------------------------------------

##Read the data and Name of the components
df=pd.read_excel('AdsorptionBreakthrough_0.1MeOH.xlsx')

#Make sure the name of the columns are the same as the component_key

zeros_row = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
df = pd.concat([zeros_row, df], ignore_index=True)

component_key=['Water','Methanol']

#Get the time and concentration values
t_vals=df['Time(hr)'].values*60 #min
cvalues_exp=df[component_key].values #[-] normalized to the feed concentration

#Upper boundary for time hr (end of breakthrough experiment)
tend=t_vals[-1]
#Scale times values and call it t_ads
t_ads=t_vals/tend 

#Density
rho=990 #kg/m^3

#Viscosity
mu=1e-3 #Pa-s

#Length of the column cm
L=20

#Diamter of the column cm
d=1.

#Bed Area
Area=np.pi*d**2/4

#Flow rate cc/min
q_cc_min=2.

#Ads density
rho_ads=.71 #g/cc

#Porosity
M_ads=6.31 #g
V_col=Area*L #cc

eb=1-(M_ads/rho_ads)/V_col

#Particle Porosity
ep=.66

#Mass Transfer
kapp=1e-1

#Dead Volume (in cc)
DV=.5 #cc

#Velocity cm/h
u_superficial=q_cc_min/Area
U=u_superficial/eb

#Feed concentration
CF_gL= np.array([910,81]) #g/L
CF= CF_gL/1000 #g/cc

#Initial guesses for Pe Weng and Chang correlation
Rp=610e-4 #cm
Re=rho*(U/100/60)*(d/100)/mu #reynold's number

Pe=L/(2*Rp)*(.18+.008*Re**.59)/eb #[-]
Dax=U*L/Pe #cm^2/min

#Initial guesses for parameters
kapp=[1.,1.] #1/min
H=[1e-3,.06] #cc(sol)/cc(MFI)
qm=[1e-3,.049*rho_ads] #g/ccMFI
K=[1e-3,79.] #cc/g
Pe=640. #[-]
Dax=U*L/Pe #cm^2/min
Dax_DV=.1 #cm^2/min

#Define dictionaries
dict_cvalues={}
dict_CF={}
dict_H={}
dict_qm={}
dict_K={}
dict_kapp={}

for j,compj in enumerate(component_key):
    
    dict_CF[compj]=CF[j]
    dict_H[compj]=H[j]
    dict_qm[compj]=qm[j]
    dict_K[compj]=K[j]
    dict_kapp[compj]=kapp[j]
    
    k=0
    
    for i in t_ads:
        dict_cvalues[compj,i]=cvalues_exp[k,j]*CF[j]
        k+=1

#Number of discretization points in time t and space x
nfet=len(t_ads)
nfex=15

#Number of collocation points if the collocation points method is used
ncp=5

#Isotherms constants
isoth='MLL'

#Dead Volume (in cc)
DV=.5 #cc
#%%
# -------------------------------------------------------------------
####Define PYOMO Model
# -------------------------------------------------------------------

#Define model parameters as variables
m=ConcreteModel()

#Define Continous set of independent variables
m.x = ContinuousSet(bounds=(0.,1.))
m.t = ContinuousSet(initialize=t_ads)
m.comp=Set(initialize=component_key)

#Define Parameters change boundaries accordingly
m.Dax= Var(initialize = Dax,bounds=(Dax/10,Dax*10))
m.Dax_DV= Var(initialize = Dax_DV,bounds=(Dax_DV/10,Dax_DV*10))
m.kapp = Var(m.comp,initialize = dict_kapp,bounds=(.1,10))
m.qm = Var(m.comp,initialize = dict_qm,bounds=(.01,.2))
m.K = Var(m.comp,initialize = dict_K,bounds=(.01,300))

if isoth=='MLL':
    m.H = Var(m.comp,initialize = dict_H,bounds=(1e-4,1))

m.Dax.fix(Dax)
m.Dax_DV.fix(Dax_DV)

for i in m.comp:
    m.kapp[i].fix(dict_kapp[i])
    m.qm[i].fix(dict_qm[i])
    m.K[i].fix(dict_K[i])
    if isoth=='MLL':
        m.H[i].fix(dict_H[i])

#Define Experimental Values
m.Cexp = Param(m.comp,m.t,initialize = dict_cvalues)

#Define Variables and other parameters
m.C = Var(m.comp,m.t,m.x)# concentration in liquid phase g/cc
m.C_dv=Var(m.comp,m.t,m.x)# dead volume concentration g/cc
m.CF = Param(m.comp,initialize=dict_CF) #feed concentrations g/cc
m.Q = Var(m.comp,m.t, m.x) #Solid concentrations g/ccMFI
m.Cp = Var(m.comp,m.t,m.x)# concentration inside the particle (intraparticle porosity, which is macropore and maybe some mesopore) 

#Define velocity as a variable
m.U = Param(initialize=U) # interstitial fluid velocity in column cm/min
m.eb = Param(initialize = eb) #[-]
m.L = Param(initialize = L) #cm

#Dead volume #Params
DV_Area=Area/5
DV_L=DV/DV_Area
DV_U=u_superficial*(Area/DV_Area)

#Define derivative variables of C and Q
m.dCdt = DerivativeVar(m.C, wrt=m.t) # differentiated liquid concentration respect to time
m.dQdt = DerivativeVar(m.Q, wrt=m.t)
m.dCdx = DerivativeVar(m.C, wrt=m.x)
m.dC2dx2 = DerivativeVar(m.C, wrt=(m.x, m.x))
m.dCpdt = DerivativeVar(m.Cp, wrt=m.t)

m.dC_dvdx = DerivativeVar(m.C_dv, wrt=m.x)
m.dC2_dvdx2 = DerivativeVar(m.C_dv, wrt=(m.x,m.x))
m.dC_dvdt = DerivativeVar(m.C_dv, wrt=m.t)

# -------------------------------------------------------------------
####Define Constraints
# -------------------------------------------------------------------

#Constraints
#Mass Balance Constraint
def MassBalanceLiquid_rule(m, i,k,l):
    if (l==0 or l==1): return Constraint.Skip
    else: return m.dCdt[i,k,l]/tend + ((1-m.eb)/m.eb)*m.kapp[i]*(m.C[i,k,l]-m.Cp[i,k,l]) + m.U*m.dCdx[i,k,l]/m.L - m.Dax*m.dC2dx2[i,k,l]/m.L/m.L == 0

#Mass Transfer Constraint, linear driving force
def MassBalanceSolid_rule(m, i,k,l):
    return (m.dQdt[i,k,l]+ep*m.dCpdt[i,k,l])/tend == m.kapp[i]*(m.C[i,k,l]-m.Cp[i,k,l])

def DV_rule(m, i,k,l):
    if l==0: return Constraint.Skip
    else: return m.dC_dvdt[i,k,l]/tend==-DV_U*m.dC_dvdx[i,k,l]/DV_L+m.Dax_DV*m.dC2_dvdx2[i,k,l]/DV_L/DV_L

#Define isotherm model
def Equilibrium_rule(m, i,k,l):
    if isoth=='Single Langmuir':
        return m.Q[i,k,l]==m.qm[i]*m.K[i]*m.Cp[i,k,l]/(1+m.K[i]*m.Cp[i,k,l])
    elif isoth=='L':
        return m.Q[i,k,l]==m.qm[i]*m.K[i]*m.Cp[i,k,l]/(1+sum(m.K[v]*m.Cp[v,k,l] for v in m.comp))
    elif isoth=='MLL':
        return m.Q[i,k,l]==m.H[i]*m.Cp[i,k,l]+m.qm[i]*m.K[i]*m.Cp[i,k,l]/(1+sum(m.K[v]*m.Cp[v,k,l] for v in m.comp))
    
#Write up the constraints
m.MassBalanceLiquid=Constraint(m.comp,m.t, m.x, rule=MassBalanceLiquid_rule)
m.MassBalanceSolid=Constraint(m.comp,m.t, m.x, rule=MassBalanceSolid_rule)
m.Equilibrium=Constraint(m.comp,m.t, m.x, rule=Equilibrium_rule)

m.DV_cons=Constraint(m.comp,m.t,m.x, rule=DV_rule)

# -------------------------------------------------------------------
####Define BC0s and IC
# -------------------------------------------------------------------

#Boundary Condition, at the entrance of the column (x=0) concentration is C0, for all t>0
#Initial Condition, C = 0 for all x
def InitialCondition_rule(m, i,l):
    return m.C[i,0.,l]==0

def InitialCondition_rule2(m, i,l):
    return m.Q[i,0.,l]==0
    
def InitialConditionCp_rule(m, i,l):
    return m.Cp[i,0.,l]==0

def BoundaryCondition_rule(m, i,k):
    if k==0: return Constraint.Skip
    else: return m.C_dv[i,k,1.]==m.C[i,k,0]-m.Dax*m.dCdx[i,k,0.]/m.L/m.U
      
def BoundaryCondition_rule2(m, i,k):
    if k==0: return Constraint.Skip
    else: return m.dCdx[i,k,1.]==0

#Write up these conditions as constraints   

m.InitialCondition=Constraint(m.comp,m.x, rule=InitialCondition_rule)
m.InitialCondition2=Constraint(m.comp,m.x, rule=InitialCondition_rule2)
m.InitialConditionCp=Constraint(m.comp,m.x, rule=InitialConditionCp_rule)
m.BoundaryCondition=Constraint(m.comp,m.t, rule=BoundaryCondition_rule)
m.BoundaryCondition2=Constraint(m.comp,m.t, rule=BoundaryCondition_rule2)

def InitialConditionDV_rule(m, i,l):
    return m.C_dv[i,0.,l]==0
    
def BoundaryConditionDV_rule(m, i,k):
    if k==0: return Constraint.Skip
    else: return m.C_dv[i,k,0.]==m.CF[i]

m.InitialConditioncstr=Constraint(m.comp,m.x, rule=InitialConditionDV_rule)
m.BoundaryConditioncstr=Constraint(m.comp,m.t, rule=BoundaryConditionDV_rule)

# -------------------------------------------------------------------
#### Discretization
# -------------------------------------------------------------------

#Set up the discretization grid, use 'Backward' for finite_difference
#Use finite difference for x and collocation for t

tscheme='BACKWARD'
xscheme='CENTRAL'

discretizet=TransformationFactory('dae.collocation')
#discretizex=TransformationFactory('dae.collocation')

#discretizet=TransformationFactory('dae.finite_difference')
discretizex=TransformationFactory('dae.finite_difference')

discretizet.apply_to(m,nfe=nfet, ncp=ncp , wrt=m.t, scheme='LAGRANGE-RADAU')
#discretizex.apply_to(m,nfe=nfex, ncp=ncp , wrt=m.x, scheme='LAGRANGE-RADAU')

#discretizet.apply_to(m,nfe=nfet,wrt=m.t,scheme=tscheme)
discretizex.apply_to(m,nfe=nfex,wrt=m.x,scheme=xscheme)

def C_AxialDerivativeConstraintBeginning_rule(m, i,k):
    return m.dCdx[i,k,0]==(-3*m.C[i,k,0]+4*m.C[i,k,m.x.card(2)]-m.C[i,k,m.x.card(3)])/2/m.x.card(2)
m.C_AxialDerivativeConstraintBeginning=Constraint(m.comp,m.t,rule=C_AxialDerivativeConstraintBeginning_rule)

def C_Axial2ndDerivativeConstraintBeginning_rule(m, i,k):
    return m.dC2dx2[i,k,0]==(2*m.C[i,k,0]-5*m.C[i,k,m.x.card(2)]+4*m.C[i,k,m.x.card(3)]-m.C[i,k,m.x.card(4)])/(m.x.card(2))**2
m.C_Axial2ndDerivativeConstraintBeginning=Constraint(m.comp,m.t,rule=C_Axial2ndDerivativeConstraintBeginning_rule)

def C_DV_Axial2ndDerivativeConstraintBeginning_rule(m, i,k):
    return m.dC2_dvdx2[i,k,0]==(2*m.C_dv[i,k,0]-5*m.C_dv[i,k,m.x.card(2)]+4*m.C_dv[i,k,m.x.card(3)]-m.C_dv[i,k,m.x.card(4)])/(m.x.card(2))**2
m.C_DV_Axial2ndDerivativeConstraintBeginning=Constraint(m.comp,m.t,rule=C_DV_Axial2ndDerivativeConstraintBeginning_rule)

def C_dv_AxialDerivativeConstraintBeginning_rule(m, i,k):
    return m.dC_dvdx[i,k,0]==(-3*m.C_dv[i,k,0]+4*m.C_dv[i,k,m.x.card(2)]-m.C_dv[i,k,m.x.card(3)])/2/m.x.card(2)
m.C_dv_AxialDerivativeConstraintBeginning=Constraint(m.comp,m.t,rule=C_dv_AxialDerivativeConstraintBeginning_rule)

if xscheme == 'CENTRAL':
    def C_AxialDerivativeConstraintEnd_rule(m, i,k):
        return m.dCdx[i,k,m.x.card(-1)]==(3*m.C[i,k,m.x.card(-1)]-4*m.C[i,k,m.x.card(-2)]+m.C[i,k,m.x.card(-3)])/2/(m.x.card(-1)-m.x.card(-2))
    m.C_AxialDerivativeConstraintEnd=Constraint(m.comp,m.t,rule=C_AxialDerivativeConstraintEnd_rule)
    def C_Axial2ndDerivativeConstraintEnd_rule(m, i,k):
        return m.dC2dx2[i,k,m.x.card(-1)]==(2*m.C[i,k,m.x.card(-1)]-5*m.C[i,k,m.x.card(-2)]+4*m.C[i,k,m.x.card(-3)]-m.C[i,k,m.x.card(-4)])/(m.x.card(-1)-m.x.card(-2))**2
    m.C_Axial2ndDerivativeConstraintEnd=Constraint(m.comp,m.t,rule=C_Axial2ndDerivativeConstraintEnd_rule)
    def C_DV_Axial2ndDerivativeConstraintEnd_rule(m, i,k):
        return m.dC2_dvdx2[i,k,m.x.card(-1)]==(2*m.C_dv[i,k,m.x.card(-1)]-5*m.C_dv[i,k,m.x.card(-2)]+4*m.C_dv[i,k,m.x.card(-3)]-m.C_dv[i,k,m.x.card(-4)])/(m.x.card(-1)-m.x.card(-2))**2
    m.C_DV_Axial2ndDerivativeConstraintEnd=Constraint(m.comp,m.t,rule=C_DV_Axial2ndDerivativeConstraintEnd_rule)
    def C_dv_AxialDerivativeConstraintEnd_rule(m, i,k):
        return m.dC_dvdx[i,k,m.x.card(-1)]==(3*m.C_dv[i,k,m.x.card(-1)]-4*m.C_dv[i,k,m.x.card(-2)]+m.C_dv[i,k,m.x.card(-3)])/2/(m.x.card(-1)-m.x.card(-2))
    m.C_dv_AxialDerivativeConstraintEnd=Constraint(m.comp,m.t,rule=C_dv_AxialDerivativeConstraintEnd_rule)
elif xscheme == 'BACKWARD':
    def C_Axial2ndDerivativeConstraintBeginning_rule2(m, i,k):
        return m.dC2dx2[i,k,m.x.card(2)]==(2*m.C[i,k,m.x.card(2)]-5*m.C[i,k,m.x.card(3)]+4*m.C[i,k,m.x.card(4)]-m.C[i,k,m.x.card(5)])/(m.x.card(2))**2
    m.C_Axial2ndDerivativeConstraintBeginning2=Constraint(m.col,m.t,rule=C_Axial2ndDerivativeConstraintBeginning_rule2)

############Simulate Breakthrough

report = build_model_size_report(m)

print('Num constraints: ', report.activated.constraints)
print('Num variables: ', report.activated.variables)
#%%
# -------------------------------------------------------------------
# Solver options
# -------------------------------------------------------------------

#### Create the ipopt_sens solver plugin using the ASL interface
solver = 'ipopt_sens'
solver_io = 'nl'
stream_solver = True    # True prints solver output to screen
keepfiles =     False    # True prints intermediate file names (.nl,.sol,...)
opt = SolverFactory(solver)#,solver_io=solver_io)
####

opt.options['mu_init'] = 1e-3
#opt.options['bound_mult_init_method']='mu-based'
#opt.options['ma57_pivtol'] = 1e-8
opt.options['max_iter'] = 5000
#opt.options['linear_system_scaling'] = 'mc19'
#
# opt.options['linear_solver'] = 'ma97'
# opt.options['linear_solver'] = 'ma57'
opt.options['linear_solver'] = 'ma27'

#m.preprocess()
results = opt.solve(m, tee=True)
#%%
####Plotting
#Collect Results for Plotting

c=len(m.comp)
r=len(m.t)

C_num=np.zeros((c,r))
t=np.zeros(r)
for i in range(r): t[i]=m.t.card(i+1)

for i in range(c):
    for j in range(r):
        C_num[i,j]=value(m.C[m.comp.card(i+1),m.t.card(j+1),m.x.card(-1)])

#Plot
plt.rcParams.update({'font.size': 14})

colors=['k','b','r']
for i in range(c):
    plt.plot(t*tend,C_num[i]/CF[i],color=colors[i],label=component_key[i])
    plt.plot(t_ads*tend,cvalues_exp.T[i],'o',color=colors[i])
    
plt.ylabel('C/CF')
plt.xlabel('Time,t (min)')

plt.legend(loc='lower right')

print('Fitted Mass Transfer Values')
for i in m.kapp:
    print('{:s} kapp = {:.2f} min^-1'.format(i,value(m.kapp[i])))

print('\nFitted Isotherm Parameters')
for i in m.comp:
    if isoth=='L' or isoth=='Single Langmuir':
        print('{:s}, qm = {:.3f} g/ccMFI, K = {:.2f} cc/g'.format(i,value(m.qm[i]),value(m.K[i])))
    else:
        print('{:s}, qm = {:.3f} g/ccMFI, K = {:.2f} cc/g, H = {:.2e} cc/ccMFI'.format(i,value(m.qm[i]),value(m.K[i]),value(m.H[i])))        

plt.title('Experimental vs. Predicted Breakthrough')# Varying kapp')
plt.show()

#%%

#Free variables for fitting comment or uncomment to free or fix variables

#m.qm.free()
#m.K.free()
m.kapp.free()
m.kapp['Water'].fix()
#if isoth=='MLL':
#    m.H.free()
#    print()
    
R=.1 #Regularization Factor

#weights for data points dictionary
dict_w={}

for i in m.comp:
    
    Ctemp=np.array([value(m.Cexp[i,ti]) for ti in t_ads])
    
    f=np.arange(len(Ctemp))[Ctemp/dict_CF[i]<1e-1][-1] #find the last zero-value concentration
    
    for k,ti in enumerate(t_ads):
        if k<f: dict_w[i,ti]=1/Ctemp[f+1]*1000
        else: dict_w[i,ti]=1/Ctemp[k]

if isoth=='MLL':
    m.obj=Objective(expr=sum(((m.C[i,k,1.]-m.Cexp[i,k])*1)**2 for i in m.comp for k in t_ads-[0.]))# + R*(sum(((m.qm[i]-dict_qm[i])/dict_qm[i])**2 + ((m.K[i]-dict_K[i])/dict_K[i])**2 + ((m.H[i]-dict_H[i])/dict_H[i])**2 +0*((m.kapp[i]-dict_kapp[i])/dict_kapp[i])**2 for i in m.comp)))
else:
    m.obj=Objective(expr=sum(((m.C[i,k,1.]-m.Cexp[i,k]))**2 for i in m.comp for k in t_ads-[0.]))# + R*(sum(((m.qm[i]-dict_qm[i])/dict_qm[i])**2 + ((m.K[i]-dict_K[i])/dict_K[i])**2 +((m.kapp[i]-dict_kapp[i])/dict_kapp[i])**2 for i in m.comp)))

# -------------------------------------------------------------------
# Solver options
# -------------------------------------------------------------------

#### Create the ipopt_sens solver plugin using the ASL interface
solver = 'ipopt_sens'
solver_io = 'nl'
stream_solver = True    # True prints solver output to screen
keepfiles =     False    # True prints intermediate file names (.nl,.sol,...)
opt = SolverFactory(solver)#,solver_io=solver_io)
####

opt.options['mu_init'] = 1e-3
#opt.options['bound_mult_init_method']='mu-based'
#opt.options['ma57_pivtol'] = 1e-8
opt.options['max_iter'] = 5000
#opt.options['linear_system_scaling'] = 'mc19'
#
# opt.options['linear_solver'] = 'ma97'
# opt.options['linear_solver'] = 'ma57'
opt.options['linear_solver'] = 'ma27'

#m.preprocess()
results = opt.solve(m, tee=True)
        
#%%
####Plotting

# Set global font properties
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 250

#Collect Results for Plotting

c=len(m.comp)
r=len(m.t)

C_num=np.zeros((c,r))
t=np.zeros(r)
for i in range(r): t[i]=m.t.card(i+1)

for i in range(c):
    for j in range(r):
        C_num[i,j]=value(m.C[m.comp.card(i+1),m.t.card(j+1),m.x.card(-1)])

#Plot
plt.rcParams.update({'font.size': 14})

colors=['b','orange']
for i in range(c):
    plt.plot(t*tend,C_num[i]/CF[i],color=colors[i],label=component_key[i])
    plt.plot(t_ads*tend,cvalues_exp.T[i],'o',color=colors[i])
    
plt.ylabel('C/CF',fontweight='bold')
plt.xlabel('Time,t (min)',fontweight='bold')

print('Fitted Mass Transfer Values')
for i in m.kapp:
    print('{:s} kapp = {:.2f} min^-1'.format(i,value(m.kapp[i])))

print('\nFitted Isotherm Parameters')
for i in m.comp:
    if isoth=='L' or isoth=='Single Langmuir':
        print('{:s}, qm = {:.3f} g/ccMFI, K = {:.2f} cc/g'.format(i,value(m.qm[i]),value(m.K[i])))
    else:
        print('{:s}, qm = {:.3f} g/ccMFI, K = {:.2f} cc/g, H = {:.2e} cc/ccMFI'.format(i,value(m.qm[i]),value(m.K[i]),value(m.H[i])))        

#plt.text(10,.1,'{:s}\nqm = {:.3f} g/ccMFI\nK = {:.2f} cc/g\nkapp = {:.2f}min^-1'.format(value(m.comp[2]),value(m.qm[m.comp[2]]),value(m.K[m.comp[2]]),value(m.kapp[m.comp[2]])))
plt.text(12,.35,'\nPe = {:.0f}\n'.format(value(m.U*m.L/m.Dax))+r'$\mathregular{k_{app,water}}$'+ ' = {:.1f} '.format(value(m.kapp['Water']))+ r'$\mathregular{min^{-1}}$'+'\n'+r'$\mathregular{k_{app,MeOH}}$'+ ' = {:.2f} '.format(value(m.kapp['Methanol']))+ r'$\mathregular{min^{-1}}$')

plt.legend(loc='upper left')

plt.show()