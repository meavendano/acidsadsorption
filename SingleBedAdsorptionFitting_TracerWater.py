# Developed and original work by Marco Avendano
# Code for the Adsorption Breakthrough of Water (tracer component)

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

component_key=['Methanol']
tracer_key=['Water']

#Get the time and concentration values
t_vals=df['Time(hr)'].values*60 #min
cvalues_exp=df[tracer_key].values #[-] normalized to the feed concentration

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
kapp=100e-1

#Dead Volume (in cc)
DV=.5 #cc

#Velocity cm/h
u_superficial=q_cc_min/Area
U=u_superficial/eb

#Feed concentration
CF_gL= np.array([910]) #g/L
CF= CF_gL/1000 #g/cc

#Initial guesses for Pe Weng and Chang correlation
Rp=610e-4 #cm
Re=rho*(U/100/60)*(d/100)/mu #reynold's number

Pe=L/(2*Rp)*(.18+.008*Re**.59)/eb #[-]
Pe=1000
Dax=U*L/Pe #cm^2/min
#%%
#Define dictionaries

dict_CF={}
dict_cvalues={}

for j,compj in enumerate(tracer_key):
    
    dict_CF=CF[j]
    
    k=0
    for i in t_ads:
        dict_cvalues[compj,i]=cvalues_exp[k,j]*CF[j]
        k+=1

#Number of discretization points in time t and space x
nfet=len(t_ads)
nfex=25

#Number of collocation points if the collocation points method is used
ncp=5
#%%
# -------------------------------------------------------------------
####Define PYOMO Model
# -------------------------------------------------------------------

#Define model parameters as variables
m=ConcreteModel()

#Define Continous set of independent variables
m.x = ContinuousSet(bounds=(0.,1.))
m.t = ContinuousSet(initialize=t_ads)
m.comp=Set(initialize=tracer_key)
#Define Parameters change boundaries accordingly
m.Dax= Var(initialize = Dax,bounds=(Dax/1000,Dax*100))
m.DV=Var(initialize=DV,bounds=(DV/10,DV*10))
m.eb = Var(initialize = eb,bounds=(eb/2,eb*2)) #[-]
m.kapp=Var(initialize=kapp,bounds=(kapp/100,kapp*100))
m.ep = Var(initialize = ep,bounds=(ep/2,ep*2)) #[-]

m.Dax.fix(Dax)
m.DV.fix(DV)
m.eb.fix(eb)
m.kapp.fix(kapp)
m.ep.fix(ep)
#%%
#Define Experimental Values
m.Cexp = Param(m.comp,m.t,initialize = dict_cvalues)

#Define Variables and other parameters
m.C = Var(m.comp,m.t,m.x)# concentration in liquid phase g/cc
m.C_dv=Var(m.comp,m.t,m.x)# dead volume concentration g/cc
m.Cp = Var(m.comp,m.t,m.x)# concentration inside the particle (intraparticle porosity, which is macropore and maybe some mesopore) 

m.CF = Param(m.comp,initialize=dict_CF) #feed concentrations g/cc

#Define velocity as a variable
m.U = Param(initialize=U) # interstitial fluid velocity in column cm/min
m.L = Param(initialize = L) #cm

#Define derivative variables of C and Q
m.dCdt = DerivativeVar(m.C, wrt=m.t) # differentiated liquid concentration respect to time
m.dCdx = DerivativeVar(m.C, wrt=m.x)
m.dC2dx2 = DerivativeVar(m.C, wrt=(m.x, m.x))
m.dCpdt = DerivativeVar(m.Cp, wrt=m.t)

m.dC_dvdx = DerivativeVar(m.C_dv, wrt=m.x)
m.dC_dvdt = DerivativeVar(m.C_dv, wrt=m.t)

# -------------------------------------------------------------------
####Define Constraints
# -------------------------------------------------------------------

#Constraints
#Mass Balance Constraint
def MassBalanceLiquid_rule(m, i,k,l):
    if (l==0 or l==1): return Constraint.Skip
    else: return m.dCdt[i,k,l]/tend + (1-m.eb)/m.eb*m.kapp*m.ep*m.dCpdt[i,k,l]/tend + m.U*m.dCdx[i,k,l]/m.L - m.Dax*m.dC2dx2[i,k,l]/m.L/m.L == 0

def MassBalanceSolid_rule(m, i,k,l):
    return m.ep*m.dCpdt[i,k,l]/tend == m.kapp*(m.C[i,k,l]-m.Cp[i,k,l])

def DV_rule(m, i,k,l):
    if l==0: return Constraint.Skip
    else: return m.dC_dvdt[i,k,l]/tend*m.DV==-q_cc_min*m.dC_dvdx[i,k,l]

#Write up the constraints
m.MassBalanceLiquid=Constraint(m.comp,m.t, m.x, rule=MassBalanceLiquid_rule)
m.MassBalanceSolid=Constraint(m.comp,m.t,m.x, rule=MassBalanceSolid_rule)

m.DV_cons=Constraint(m.comp,m.t,m.x, rule=DV_rule)

# -------------------------------------------------------------------
####Define BC0s and IC
# -------------------------------------------------------------------

#Boundary Condition, at the entrance of the column (x=0) concentration is C0, for all t>0
#Initial Condition, C = 0 for all x
def InitialCondition_rule(m, i,l):
    return m.C[i,0.,l]==0

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

colors=['r','yellow','magenta']
for i in range(c):
    plt.plot(t*tend,C_num[i]/CF[i],color=colors[i],label=tracer_key[i])
    plt.plot(t_ads*tend,cvalues_exp.T[i],'o',color=colors[i])
    
plt.ylabel('C/CF')
plt.xlabel('Time,t (min)')

plt.legend(loc='lower right')

plt.text(10.,.55,'\nPe = {:.1f}\nDV = {:.2f} cc\neb= {:.2f}'.format(value(m.U*m.L/m.Dax),value(m.DV),value(m.eb)))    
plt.title('Experimental vs. Predicted Tracer Breakthrough')# Varying kapp')
plt.show()

#%%

#Free variables for fitting comment or uncomment to free or fix variables
m.Dax.free()
#m.DV.free()
m.kapp.free()
#m.eb.free()
#m.ep.free()

R=.01 #Regularization Factor
f=np.arange(len(cvalues_exp))[cvalues_exp.flatten()<1e-1][-1] #find the last zero-value concentration

#weights for data points
w=np.zeros(len(cvalues_exp))

w[:f+1]=1/cvalues_exp[f+1]
w[f+1:]=1/cvalues_exp[f+1:].flatten()

#dictionary of weights

dict_w={}
for k,ti in enumerate(t_ads):
    dict_w[ti]=w[k]*1000

m.obj=Objective(expr=sum((m.C[i,k,1.]-m.Cexp[i,k])**2*dict_w[k] for i in m.comp for k in t_ads-[0.]))# + R*(1e3*((m.eb-eb)/eb)**2 + 10*((m.DV-DV)/DV)**2 + ((m.Dax-Dax)/Dax)**2)+1e-5*((m.kapp-kapp)/kapp)**2+1e-1*((m.ep-ep)/ep)**2)

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

# Set global font properties
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 250

C_num=np.zeros((c,r))
t=np.zeros(r)
for i in range(r): t[i]=m.t.card(i+1)

for i in range(c):
    for j in range(r):
        C_num[i,j]=value(m.C[m.comp.card(i+1),m.t.card(j+1),m.x.card(-1)])

#Plot
plt.rcParams.update({'font.size': 14})

colors=['blue']
for i in range(c):
    plt.plot(t*tend,C_num[i]/CF[i],color=colors[i],label=tracer_key[i])
    plt.plot(t_ads*tend,cvalues_exp.T[i],'o',color=colors[i])
    
plt.ylabel('C/CF',fontweight='bold')
plt.xlabel('Time,t (min)',fontweight='bold')

#plt.text(10.,.55,'\nPe = {:.1f}\nDV = {:.2f} cc\neb= {:.2f}\nep= {:.2f}\nkapp= {:.2f}min^-1'.format(value(m.U*m.L/m.Dax),value(m.DV),value(m.eb),value(m.ep),value(m.kapp)))    
plt.text(10.,.55,'\nPe = {:.1f}\n'.format(value(m.U*m.L/m.Dax))+r'$\mathregular{k_{app,water}}$'+ '= {:.2f} '.format(value(m.kapp))+ r'$\mathregular{min^{-1}}$')
plt.show()