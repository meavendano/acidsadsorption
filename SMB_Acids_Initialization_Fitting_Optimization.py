# Developed by Marco Avendano in Dec 2022 based on the original work of Kensuke Suzuki in 2020,Suriya Arulselvan in Dec 2015
# The method used in this file is the same as study by Kawajiri and Biegler
# Bibliographic information: Kawajiri Y, Biegler LT, Optimization Strategies for Simulated Moving Bed and PowerFeed Processes. AIChE Journal. 2006;52:1343-1350.

from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd

#%%
# -------------------------------------------------------------------
#Read in Data
# -------------------------------------------------------------------

##Please see the excel file for more information

filename='AcidsExpData_2023.xlsx'
df_conc=pd.read_excel(filename,sheet_name='Exp Concentration g_L',index_col=[0,1]) #experimental weight fractions
df_exp_wt=pd.read_excel(filename,sheet_name='Experimental Fractions',index_col=[0,1]) #experimental weight fractions
df_conc_ex=pd.read_excel(filename,sheet_name='Experimental Extract',index_col=0) #experimental weight fractions
df_conc_raff=pd.read_excel(filename,sheet_name='Experimental Raff',index_col=0) #experimental weight fractions
df_flow_rates=pd.read_excel(filename,sheet_name='Flow Rates cc_min',index_col=[0,1],header=0) #experimental flow rates
df_properties=pd.read_excel(filename,sheet_name='Properties',index_col=0,header=0) #experimental flow rates

exp_conc=df_conc.stack().to_dict() #experimental concentrations
exp_wt=df_exp_wt.stack().to_dict() #experimental weight fractions

exp_conc_ex=df_conc_ex.stack().to_dict() #experimental concentrations
exp_conc_raff=df_conc_raff.stack().to_dict() #experimental weight fractions

flow_rates=df_flow_rates.to_dict()[df_flow_rates.columns[0]]
     
#%%
# -------------------------------------------------------------------
####Define Parameters and Constants Section
# -------------------------------------------------------------------
 
#Number of discretization points in x
nfex=10
nfet=3

#Number of collocation points if the collocation points method is used
ncp=3

#SMB Configuration
nc=[2,2,2,2]

#Select the experimental runs to solve
data_index=['a','b','c']

#Data points
data=len(data_index)

#Total Number of Columns
ncols=sum(nc)

#Name of components (the desorbent is the last component)
#comps=df_exp_wt.columns
comps=['MA','GA','WATER','MEOH']

#Number of components
ncomp=len(comps)

#Length of the column cm
L=20

#Step time min
tstep=10

#Peclet number vL/Dax
Pe=700.

#Diameter cm
d=1.

#Bed Porosity
eb=0.44

#Particle porosity (space inside the particle that is non-adsorbed)
ep=.66

#Area cm^2
Area=np.pi*d**2/4

#Densities and feed

dict_rho={}
dict_CF={}
dict_CD={}
dict_kapp={}
dict_qm={}
dict_H={}
dict_K={}

for compi in comps:
    dict_rho[compi]=df_properties[compi]['rho']
    dict_CF[compi]=df_properties[compi]['feed']
    dict_CD[compi]=df_properties[compi]['desorb']
    dict_kapp[compi]=df_properties[compi]['kapp']
    dict_qm[compi]=df_properties[compi]['qm']/.71
    dict_K[compi]=df_properties[compi]['K']
    dict_H[compi]=df_properties[compi]['H']

    #dict_H={'GA':.999,'MEOH':.969,'WATER':.9427}    
    dict_kapp={'GA': 0.8, 'MA': 1.22, 'WATER': 1.0, 'MEOH': 0.69}
    
dict_exp_wt={}
dict_exp_conc={}
dict_exp_conc_ex={}
dict_exp_conc_raff={}
dict_U={}
dict_Dax={}

dict_UF={}
dict_UD={}
dict_UE={}
dict_UR={}
dict_tstep={}

for i in exp_wt:
    di=i[0]
    if di in data_index: 
        dict_exp_wt[i]=exp_wt[i]

        dict_exp_conc[i]=exp_wt[i]/sum(exp_wt[di,i[1],v]/dict_rho[v] for v in comps)
        
        dict_exp_conc_ex[i[0],i[-1]]=exp_conc_ex[i[0],i[-1]]
        dict_exp_conc_raff[i[0],i[-1]]=exp_conc_raff[i[0],i[-1]]
        
        dict_UF[di]=(flow_rates[di,3]-flow_rates[di,2])/Area/eb
        dict_UD[di]=(flow_rates[di,1]-flow_rates[di,4])/Area/eb
        dict_UE[di]=(flow_rates[di,1]-flow_rates[di,2])/Area/eb
        dict_UR[di]=(flow_rates[di,3]-flow_rates[di,4])/Area/eb
        
        dict_tstep[di]=tstep
                
        count=1
        for zone,nci in enumerate(nc):
            for j in range(nci): 
                dict_U[di,count]=flow_rates[di,zone+1]/Area/eb
                dict_Dax[di,count]=dict_U[di,count]*L/Pe
                count+=1

#Number of sections
nsec=4
    
isoth='MLL' #L Langmuir and MLL Mixed Linear+Langmuir

cstr=1

m=ConcreteModel()

#Define Continous set of independent variables
m.x = ContinuousSet(bounds=(0.,1.))
m.t=ContinuousSet(bounds=(0.,1.))
m.col=RangeSet(ncols)
m.comp=Set(initialize=comps)
m.data=Set(initialize=data_index)
m.cstr=RangeSet(cstr)
#Define model parameters as variables

m.L = Param(initialize = L)
m.Dax= Param(m.data,m.col,initialize = dict_Dax)
m.eb = Param(initialize = eb)
m.ep=Var(initialize=ep)
m.ep.fix(ep)

m.kapp = Var(m.comp,initialize = dict_kapp,bounds=(1e-7,100))
m.qm = Var(m.comp,initialize = dict_qm,bounds=(1e-7,.4))
m.K = Var(m.comp,initialize = dict_K,bounds=(1e-7,1e4))

if isoth=='MLL':
    m.H = Var(m.comp,initialize = dict_H,bounds=(1e-7,1))
    
for i in m.comp:
    m.kapp[i].fix(dict_kapp[i])
    m.qm[i].fix(dict_qm[i])
    m.K[i].fix(dict_K[i])
    if isoth=='MLL':
        m.H[i].fix(dict_H[i])

DV=Area*L*.026101
m.DV=Var(initialize=DV)
m.DV.fix(DV)

DVcstr=2.5
m.DVcstr=Var(initialize=DVcstr)
m.DVcstr.fix(DVcstr)


#Define Variables and other parameters
m.C = Var(m.data,m.col,m.comp,m.t,m.x, within=NonNegativeReals)# concentration in the bulk liquid phase inside the bed voidage (interparticle)
m.Q = Var(m.data,m.col,m.comp,m.t,m.x, within=NonNegativeReals)# concentration of the adsorbed phase (micropores/intracrystalline)
m.Cp = Var(m.data,m.col,m.comp,m.t,m.x)# concentration inside the particle (intraparticle porosity, which is macropore and maybe some mesopore) 
m.Cdv=Var(m.data,m.col,m.comp,m.t,m.x)
m.Ccstr=Var(m.data,m.comp,m.t,m.cstr)

m.CF = Param(m.comp,initialize=dict_CF) #concentration of the feed
m.CD = Param(m.comp,initialize=dict_CD) #concentration of the desorbent
   
#Define velocity as a variable
m.U = Var(m.data,m.col,within=PositiveReals, initialize=dict_U,bounds=(0.,30.)) # fluid velocity in column

m.UF = Var(m.data,initialize=dict_UF,bounds=(0.,30.)) #Intersitital feed velocity
m.UD = Var(m.data,initialize=dict_UD,bounds=(0.,30.)) #Intersitital desorbent velocity
m.UE = Var(m.data,initialize=dict_UE,bounds=(0.,30.)) #Intersitital extract velocity
m.UR = Var(m.data,initialize=dict_UR,bounds=(0.,30.)) #Intersitital raffinate velocity

for di in m.data:
    #Need to fix three velocities (in this case I picked feed, desorbent and extract)
    m.UF[di].fix(dict_UF[di])
    m.UD[di].fix(dict_UD[di])
    m.UE[di].fix(dict_UE[di])

#Also need to fix the solid velocity
m.tstep=Param(m.data,initialize=dict_tstep)#,bounds=(u_f*5,u_f*70))

#Define derivative variables of C, Cp and Q
m.dCdx = DerivativeVar(m.C, wrt=m.x)
m.dC2dx2 = DerivativeVar(m.C, wrt=(m.x, m.x))
m.dQdt = DerivativeVar(m.Q, wrt=m.t)
m.dCdt = DerivativeVar(m.C, wrt=m.t)
m.dCpdt = DerivativeVar(m.Cp, wrt=m.t)

m.dCdvdx = DerivativeVar(m.Cdv, wrt=m.x)
m.dCdv2dx2 = DerivativeVar(m.Cdv, wrt=(m.x, m.x))
m.dCdvdt = DerivativeVar(m.Cdv, wrt=m.t)

m.dCcstrdt=DerivativeVar(m.Ccstr,wrt=m.t)
# -------------------------------------------------------------------
####Define Constraints
# -------------------------------------------------------------------

#Constraints
#Mass Balance Constraint
def MassBalanceLiquid_rule(m, d,i,j,k,l):
    if l==0: return Constraint.Skip
    else: return m.dCdt[d,i,j,k,l]/m.tstep[d]+m.U[d,i]*m.dCdx[d,i,j,k,l]/m.L + (1-m.eb)/m.eb*m.kapp[j]*(m.C[d,i,j,k,l]-m.Cp[d,i,j,k,l]) - m.Dax[d,i]*m.dC2dx2[d,i,j,k,l]/m.L/m.L== 0

#Mass Transfer Constraint, linear driving force
def MassBalanceSolid_rule(m, d,i,j,k,l):
    return (.7*m.dQdt[d,i,j,k,l]+m.ep*m.dCpdt[d,i,j,k,l])/m.tstep[d] == m.kapp[j]*(m.C[d,i,j,k,l]-m.Cp[d,i,j,k,l])

#Define isotherm model
def Equilibrium_rule(m, d,i,j,k,l):
    if isoth=='L': return m.Q[d,i,j,k,l]==m.qm[j]*m.K[j]*m.Cp[d,i,j,k,l]/(1+sum(m.K[v]*m.Cp[d,i,v,k,l] for v in m.comp))
    elif  isoth=='MLL': return m.Q[d,i,j,k,l]==(m.H[j]*m.Cp[d,i,j,k,l])+m.qm[j]*m.K[j]*m.Cp[d,i,j,k,l]/(1+sum(m.K[v]*m.Cp[d,i,v,k,l] for v in m.comp))

#Write up the constraints
m.MassBalanceLiquid=Constraint(m.data,m.col,m.comp,m.t,m.x, rule=MassBalanceLiquid_rule)
m.MassBalanceSolid=Constraint(m.data,m.col,m.comp,m.t,m.x, rule=MassBalanceSolid_rule)
m.Equilibrium=Constraint(m.data,m.col,m.comp,m.t,m.x, rule=Equilibrium_rule)

#Flow Condition, assume all areas are the same

def MassBalanceLiquidDV_rule(m, d,i,j,k,l):
    if l==0: return Constraint.Skip
    else: return m.dCdvdt[d,i,j,k,l]/m.tstep[d]*m.DV**2+(Area*m.U[d,i]*m.eb)*m.DV*m.dCdvdx[d,i,j,k,l] - m.Dax[d,i]*Area**2/10000*m.dCdv2dx2[d,i,j,k,l]== 0

m.MassBalanceLiquidDV=Constraint(m.data,m.col,m.comp,m.t,m.x, rule=MassBalanceLiquidDV_rule)

def MassBalanceLiquidcstr_rule(m, d,j,k,c):
    return m.DVcstr/(Area*m.U[d,ncols]*eb)*m.dCcstrdt[d,j,k,c]/m.tstep[d]==m.Cdv[d,ncols,j,k,1.]-m.Ccstr[d,j,k,c]

m.MassBalanceLiquidcstr=Constraint(m.data,m.comp,m.t,m.cstr, rule=MassBalanceLiquidcstr_rule)

def FlowCondition_rule(m, d,i):
    if i==1: return m.U[d,ncols]+m.UD[d]==m.U[d,i]
    elif i==(nc[0]+1): return m.U[d,i-1]==m.U[d,i]+m.UE[d]
    elif i==(nc[0]+nc[1]+1): return m.U[d,i-1]+m.UF[d]==m.U[d,i]
    elif i==(nc[0]+nc[1]+nc[2]+1): return m.U[d,i-1]==m.U[d,i]+m.UR[d]
    else: return m.U[d,i-1]==m.U[d,i]

m.FlowCondition=Constraint(m.data,m.col, rule=FlowCondition_rule)

for di in data_index:
    m.U[di,ncols].fix(dict_U[di,ncols])
# -------------------------------------------------------------------
#### Discretization
# -------------------------------------------------------------------

#Set up the discretization grid, use 'Backward' for finite_difference
#Use finite difference for x and collocation for t

tscheme='BACKWARD'
xscheme='CENTRAL'

discretizet=TransformationFactory('dae.collocation')
discretizet.apply_to(m,nfe=nfet, ncp=ncp , wrt=m.t, scheme='LAGRANGE-RADAU')

discretizex=TransformationFactory('dae.finite_difference')
discretizex.apply_to(m,nfe=nfex,wrt=m.x,scheme=xscheme)

#Define the derivatives at the beginning and end (Pyomo doesn't do this automatically yet so it has to be done manually)

def C_AxialDerivativeConstraintBeginning_rule(m, d,i,j,k):
    return m.dCdx[d,i,j,k,0]==(-3*m.C[d,i,j,k,0]+4*m.C[d,i,j,k,m.x.card(2)]-m.C[d,i,j,k,m.x.card(3)])/2/m.x.card(2)
m.C_AxialDerivativeConstraintBeginning=Constraint(m.data,m.col,m.comp,m.t, rule=C_AxialDerivativeConstraintBeginning_rule)

def C_Axial2ndDerivativeConstraintBeginning_rule(m, d,i,j,k):
    return m.dC2dx2[d,i,j,k,0]==(2*m.C[d,i,j,k,0]-5*m.C[d,i,j,k,m.x.card(2)]+4*m.C[d,i,j,k,m.x.card(3)]-m.C[d,i,j,k,m.x.card(4)])/(m.x.card(2))**2
m.C_Axial2ndDerivativeConstraintBeginning=Constraint(m.data,m.col,m.comp,m.t, rule=C_Axial2ndDerivativeConstraintBeginning_rule)

def Cdv_AxialDerivativeConstraintBeginning_rule(m, d,i,j,k):
    return m.dCdvdx[d,i,j,k,0]==(-3*m.Cdv[d,i,j,k,0]+4*m.Cdv[d,i,j,k,m.x.card(2)]-m.Cdv[d,i,j,k,m.x.card(3)])/2/m.x.card(2)
m.Cdv_AxialDerivativeConstraintBeginning=Constraint(m.data,m.col,m.comp,m.t, rule=Cdv_AxialDerivativeConstraintBeginning_rule)

def Cdv_Axial2ndDerivativeConstraintBeginning_rule(m, d,i,j,k):
    return m.dCdv2dx2[d,i,j,k,0]==(2*m.Cdv[d,i,j,k,0]-5*m.Cdv[d,i,j,k,m.x.card(2)]+4*m.Cdv[d,i,j,k,m.x.card(3)]-m.Cdv[d,i,j,k,m.x.card(4)])/(m.x.card(2))**2
m.Cdv_Axial2ndDerivativeConstraintBeginning=Constraint(m.data,m.col,m.comp,m.t, rule=Cdv_Axial2ndDerivativeConstraintBeginning_rule)

if xscheme == 'CENTRAL':
    def C_AxialDerivativeConstraintEnd_rule(m, d,i,j,k):
        return m.dCdx[d,i,j,k,m.x.card(-1)]==(3*m.C[d,i,j,k,m.x.card(-1)]-4*m.C[d,i,j,k,m.x.card(-2)]+m.C[d,i,j,k,m.x.card(-3)])/2/(m.x.card(-1)-m.x.card(-2))
    m.C_AxialDerivativeConstraintEnd=Constraint(m.data,m.col,m.comp,m.t, rule=C_AxialDerivativeConstraintEnd_rule)
    def C_Axial2ndDerivativeConstraintEnd_rule(m, d,i,j,k):
        return m.dC2dx2[d,i,j,k,m.x.card(-1)]==(2*m.C[d,i,j,k,m.x.card(-1)]-5*m.C[d,i,j,k,m.x.card(-2)]+4*m.C[d,i,j,k,m.x.card(-3)]-m.C[d,i,j,k,m.x.card(-4)])/(m.x.card(-1)-m.x.card(-2))**2
    m.C_Axial2ndDerivativeConstraintEnd=Constraint(m.data,m.col,m.comp,m.t, rule=C_Axial2ndDerivativeConstraintEnd_rule)
    
    def Cdv_AxialDerivativeConstraintEnd_rule(m, d,i,j,k):
        return m.dCdvdx[d,i,j,k,m.x.card(-1)]==(3*m.Cdv[d,i,j,k,m.x.card(-1)]-4*m.Cdv[d,i,j,k,m.x.card(-2)]+m.Cdv[d,i,j,k,m.x.card(-3)])/2/(m.x.card(-1)-m.x.card(-2))
    m.Cdv_AxialDerivativeConstraintEnd=Constraint(m.data,m.col,m.comp,m.t, rule=Cdv_AxialDerivativeConstraintEnd_rule)
    def Cdv_Axial2ndDerivativeConstraintEnd_rule(m, d,i,j,k):
        return m.dCdv2dx2[d,i,j,k,m.x.card(-1)]==(2*m.Cdv[d,i,j,k,m.x.card(-1)]-5*m.Cdv[d,i,j,k,m.x.card(-2)]+4*m.Cdv[d,i,j,k,m.x.card(-3)]-m.Cdv[d,i,j,k,m.x.card(-4)])/(m.x.card(-1)-m.x.card(-2))**2
    m.Cdv_Axial2ndDerivativeConstraintEnd=Constraint(m.data,m.col,m.comp,m.t, rule=Cdv_Axial2ndDerivativeConstraintEnd_rule)
elif xscheme == 'BACKWARD':
    def C_Axial2ndDerivativeConstraintBeginning_rule2(m, d,i,j,k):
        return m.dC2dx2[d,i,j,k,m.x.card(2)]==(2*m.C[d,i,j,k,m.x.card(2)]-5*m.C[d,i,j,k,m.x.card(3)]+4*m.C[d,i,j,k,m.x.card(4)]-m.C[d,i,j,k,m.x.card(5)])/(m.x.card(2))**2
    m.C_Axial2ndDerivativeConstraintBeginning2=Constraint(m.data,m.col,m.comp,m.t, rule=C_Axial2ndDerivativeConstraintBeginning_rule2)

#Boundary Condition, mass balance

m.C0=Var(m.data,m.col,m.comp,m.t)

def MassBalance_rule(m, d,i,j,k):
    if k==0: return Constraint.Skip
    else:     
        if i==1: return m.Ccstr[d,j,k,cstr]*m.U[d,ncols]+m.CD[j]*m.UD[d]==m.C0[d,i,j,k]*m.U[d,i]
        elif i==(nc[0]+nc[1]+1): return m.Cdv[d,i-1,j,k,1.]*m.U[d,i-1]+m.CF[j]*m.UF[d]==m.C0[d,i,j,k]*m.U[d,i]
        else: return m.Cdv[d,i-1,j,k,1.]==m.C0[d,i,j,k]
        
def BoundaryConditionC_rule(m, d,i,j,k):
    return m.C0[d,i,j,k]==m.C[d,i,j,k,0]-m.Dax[d,i]/m.U[d,i]*m.dCdx[d,i,j,k,0]/m.L

def BoundaryConditionCdv_rule(m, d,i,j,k):
    if k==0: return Constraint.Skip
    else: return m.C[d,i,j,k,1]*m.DV==m.Cdv[d,i,j,k,0]*m.DV-(m.Dax[d,i]*Area)/10000/m.U[d,i]*m.dCdvdx[d,i,j,k,0]
        
#Write up these conditions as constraints   

m.MassBalance=Constraint(m.data,m.col,m.comp,m.t, rule=MassBalance_rule)
m.BoundaryConditionC=Constraint(m.data,m.col,m.comp,m.t, rule=BoundaryConditionC_rule)
m.BoundaryConditionCdv=Constraint(m.data,m.col,m.comp,m.t, rule=BoundaryConditionCdv_rule)

def CSSC_rule(m, d,i,j,l):
    if i==ncols: return m.C[d,ncols,j,0,l]==m.C[d,1,j,1,l]
    else: return m.C[d,i,j,0,l]==m.C[d,i+1,j,1,l]
    
def CSSCp_rule(m, d,i,j,l):
    if i==ncols: return m.Cp[d,ncols,j,0,l]==m.Cp[d,1,j,1,l]
    else: return m.Cp[d,i,j,0,l]==m.Cp[d,i+1,j,1,l]
    
def CSSQ_rule(m, d,i,j,l):
    if i==ncols: return m.Q[d,ncols,j,0,l]==m.Q[d,1,j,1,l]
    else: return m.Q[d,i,j,0,l]==m.Q[d,i+1,j,1,l]

m.CSSC=Constraint(m.data,m.col,m.comp,m.x, rule=CSSC_rule)
m.CSSCp=Constraint(m.data,m.col,m.comp,m.x, rule=CSSCp_rule)
m.CSSQ=Constraint(m.data,m.col,m.comp,m.x, rule=CSSQ_rule)
 
def CSSCdv_rule(m, d,i,j,l):
    if i==ncols: return m.Cdv[d,ncols,j,0,l]==m.Cdv[d,1,j,1,l]
    else: return m.Cdv[d,i,j,0,l]==m.Cdv[d,i+1,j,1,l]
    
m.CSSCdv=Constraint(m.data,m.col,m.comp,m.x, rule=CSSCdv_rule)

def CSScstr_rule(m, d,j):
    return m.Ccstr[d,j,0,cstr]==m.C0[d,1,j,1]
    
m.CSScstr=Constraint(m.data,m.comp, rule=CSScstr_rule)
#Define extract and raffinate velocity
# m.CE=Var(m.data,m.comp)
# m.CR=Var(m.data,m.comp)
# m.CRecy=Var(m.data,m.comp)

# def CE_rule(m, d,j):
#     return m.CE[d,j]==sum(m.C[d,nc[0],j,k,1.] for k in m.t)/len(m.t)

# def CR_rule(m, d,j):
#     return m.CR[d,j]==sum(m.C[d,nc[0]+nc[1]+nc[2],j,k,1.] for k in m.t)/len(m.t)

# def CRecy_rule(m, d,j):
#     return m.CRecy[d,j]==sum(m.C[d,ncols,j,k,1.] for k in m.t)/len(m.t)
        
# m.CE_cons=Constraint(m.data,m.comp, rule=CE_rule)
# m.CR_cons=Constraint(m.data,m.comp, rule=CR_rule)
# m.CRecy_cons=Constraint(m.data,m.comp, rule=CRecy_rule)

def CE_rule(m,d,j,k):
    return m.C[d,nc[0],j,k,1]
m.CE = Integral(m.data,m.comp,m.t,wrt=m.t,rule=CE_rule)

def CR_rule(m,d,j,k):
    return m.C[d,nc[0]+nc[1]+nc[2],j,k,1]
m.CR = Integral(m.data,m.comp,m.t,wrt=m.t,rule=CR_rule)

def Crecy_rule(m,d,j,k):
    return m.C[d,ncols,j,k,1]
m.Crecy = Integral(m.data,m.comp,m.t,wrt=m.t,rule=Crecy_rule)

def Cplot_rule(m,d,i,j,k,l):
    return m.C[d,i,j,k,l]
m.Cplot = Integral(m.data,m.col,m.comp,m.t,m.x,wrt=m.t,rule=Cplot_rule)
        
UB=1.1
LB=.9

def Ex1_rule(m, d,j):
    return m.CE[d,j]<=exp_conc_ex[d,j]/1000*UB

def Raff1_rule(m, d,j):
    return m.CR[d,j]<=exp_conc_raff[d,j]/1000*UB

def Ex2_rule(m, d,j):
    return m.CE[d,j]>=exp_conc_ex[d,j]/1000*LB

def Raff2_rule(m, d,j):
    return m.CR[d,j]>=exp_conc_raff[d,j]/1000*LB
        
m.Ex1_cons=Constraint(m.data,m.comp-['MEOH'], rule=Ex1_rule)
m.Raff1_cons=Constraint(m.data,m.comp-['WATER','MEOH'], rule=Raff1_rule)
m.Ex2_cons=Constraint(m.data,m.comp-['MEOH'], rule=Ex2_rule)
m.Raff2_cons=Constraint(m.data,m.comp-['WATER','MEOH'], rule=Raff2_rule)

# UB_etoh=1.1
# LB_etoh=.9

# def Ex1etoh_rule(m, d):
#     return m.CE[d,3]<=dict_Ex[d,3]/1000*UB_etoh

# def Ex2etoh_rule(m, d):
#     return m.CE[d,3]>=dict_Ex[d,3]/1000*LB_etoh

# m.Ex1_etoh=Constraint(m.data, rule=Ex1etoh_rule)
# m.Ex2_etoh=Constraint(m.data, rule=Ex2etoh_rule)

#m.Ex1_etoh.deactivate()
#m.Ex2_etoh.deactivate()
m.Ex1_cons.deactivate()
m.Raff1_cons.deactivate()
m.Ex2_cons.deactivate()
m.Raff2_cons.deactivate()

def Recy_rule(m, d,j):
    return m.Crecy[d,j]<=1e-2

m.Recy_cons=Constraint(m.data,['WATER'], rule=Recy_rule)

m.Recy_cons.deactivate()

from pyomo.util.model_size import build_model_size_report
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

opt.options['mu_init'] = 1e-5
#opt.options['bound_mult_init_method']='mu-based'
#opt.options['ma57_pivtol'] = 1e-8
opt.options['max_iter'] = 5000
#opt.options['linear_system_scaling'] = 'mc19'
#
#opt.options['linear_solver'] = 'ma97'
#opt.options['linear_solver'] = 'ma57'
#opt.options['linear_solver'] = 'ma27'
#opt.options['acceptable_tol']=1e-4
#m.preprocess()
results = opt.solve(m, tee=True)

#%%
print('Purity')
for j in comps[:-2]:
    print(j)
    for i in m.data:
        print('Experimental:{:.1%} Predicted:{:.1%}'.format(exp_conc_ex[i,j]/sum(exp_conc_ex[i,v] for v in m.comp-['MEOH']),value(m.CE[i,j]/sum(m.CE[i,v] for v in m.comp-['MEOH']))))
        
print('\nRecovery')
for j in comps[:-2]:
    print(j)
    for i in m.data:
        print('Experimental:{:.1%} Predicted:{:.1%}'.format(value(exp_conc_ex[i,j]/1000*m.UE[i]/(m.CF[j]*m.UF[i])),value(m.CE[i,j]*m.UE[i]/(m.CF[j]*m.UF[i]))))
#%%
print('Purity')
for i in m.data:
    print('Experimental:{:.1%} Predicted:{:.1%}'.format(sum(exp_conc_ex[i,v] for v in m.comp-['MEOH','WATER'])/sum(exp_conc_ex[i,v] for v in m.comp-['MEOH']),value(sum(m.CE[i,v] for v in m.comp-['MEOH','WATER'])/sum(m.CE[i,v] for v in m.comp-['MEOH']))))

print('\nRecovery')
for i in m.data:
    print('Experimental:{:.1%} Predicted:{:.1%}'.format(value(sum(exp_conc_ex[i,v]/1000 for v in m.comp-['MEOH','WATER'])*m.UE[i]/(sum(m.CF[v] for v in m.comp-['MEOH','WATER'])*m.UF[i])),value(sum(m.CE[i,v] for v in m.comp-['MEOH','WATER'])*m.UE[i]/(sum(m.CF[v] for v in m.comp-['MEOH','WATER'])*m.UF[i]))))

   #%%

data_plot=[3]#[1,2,3]
  
if len(data_plot)==1:
    fig,axes =plt.subplots(len(data_plot),1,sharex=True)
    axes=[axes]
else:
    fig,axes =plt.subplots(len(data_plot),1,sharex=True,figsize=(5,7))

colors=['r','b','g'][:len(comps)]
dict_colors={}
for i,ci in enumerate(colors):
    dict_colors[comps[i]]=ci

r=len(m.t)
t=np.zeros(r)
for i in range(r): t[i]=m.t.card(i+1)

c=len(m.x)
x=np.zeros(c*ncols)
#for i in range(c): x[i]=m.x.card(i+1)

for di,datai in enumerate(data_plot):
    
    C_num=np.zeros((ncomp,c*ncols))
    C_exp=np.zeros((ncomp,ncols))
    
    for j,compj in enumerate(comps):
        count=0
        for i in range(ncols):
            for l in range(c):
                C_num[j,count]=value(m.C[datai,i+1,compj,m.t.card(-1),m.x.card(l+1)])
                x[count]=count
                count+=1
                C_exp[j,i]=dict_exp_wt[datai,i+1,compj]
                
    for j,compj in enumerate(comps):
        axes[di].plot(x,C_num[j]/sum(C_num)*100,c=dict_colors[compj],label=compj)
        axes[di].plot(x[c-1::c],C_exp[j]*100,'o',c=dict_colors[compj])
        
    axes[di].set_xticks(np.linspace(0,count,9), [str(int(i)) for i in np.linspace(0,L*ncols,9)])
    axes[di].set_ylabel('wt%')
    axes[di].set_xlabel('SMB Length (cm)')
    
    title_string='SMB Run ' + str(datai)
    
    plt.title(title_string)
    
    plt.legend(loc='center left')
    axes[di].set_ylim((0,100))
    
    # s=['D','E','F','R']
    # a=['->','<-','->','<-']
    # nc_string=[0,nc[0],nc[0]+nc[1],nc[0]+nc[1]+nc[2]]
    
    # for i in range(nsec): 
    #     axes[di].annotate(s[i], xy=((nc_string[i])/ncols*len(x), 0),xytext=((nc_string[i])/ncols*len(x), 20) ,horizontalalignment="center",
    #     arrowprops=dict(arrowstyle=a[i],lw=1),fontsize=14)

plt.show()

   #%%

data_plot=data_index#[1,2,3]
  
if len(data_plot)==1:
    fig,axes =plt.subplots(len(data_plot),1,sharex=True)
    axes=[axes]
else:
    fig,axes =plt.subplots(len(data_plot),1,sharex=True,figsize=(5,7))

colors=['r','b','g'][:len(comps)]
dict_colors={}
for i,ci in enumerate(colors):
    dict_colors[comps[i]]=ci

r=len(m.t)
t=np.zeros(r)
for i in range(r): t[i]=m.t.card(i+1)

c=len(m.x)
x=np.zeros(c*ncols)
#for i in range(c): x[i]=m.x.card(i+1)

for di,datai in enumerate(data_plot):
    
    C_num=np.zeros((ncomp,c*ncols))
    C_exp=np.zeros((ncomp,ncols))
    
    for j,compj in enumerate(comps):
        count=0
        for i in range(ncols):
            for l in range(c):
                C_num[j,count]=value(m.C[datai,i+1,compj,m.t.card(-1),m.x.card(l+1)])
                x[count]=count
                count+=1
                C_exp[j,i]=dict_exp_conc[datai,i+1,compj]
                
    for j,compj in enumerate(comps):
        axes[di].plot(x,C_num[j]*1000,c=dict_colors[compj],label=compj)
        axes[di].plot(x[c-1::c],C_exp[j]*1000,'o',c=dict_colors[compj])
        
    axes[di].set_xticks(np.linspace(0,count,9), [str(int(i)) for i in np.linspace(0,L*ncols,9)])
    axes[di].set_ylabel('g/L')
    axes[di].set_xlabel('SMB Length (cm)')
    
    title_string='SMB Run ' + str(datai)
    
    plt.title(title_string)
    
    plt.legend(loc='center left')
    axes[di].set_ylim((0,1000))
    
    s=['D','E','F','R']
    a=['->','<-','->','<-']
    nc_string=[0,nc[0],nc[0]+nc[1],nc[0]+nc[1]+nc[2]]
    
    for i in range(nsec): 
        axes[di].annotate(s[i], xy=((nc_string[i])/ncols*len(x), 0),xytext=((nc_string[i])/ncols*len(x), 200) ,horizontalalignment="center",
        arrowprops=dict(arrowstyle=a[i],lw=1),fontsize=14)

plt.show()

#%%
##Minimization Function

for i in m.comp:
    m.kapp.free()
    #m.kapp['WATER'].free()
    #m.kapp['GA'].free()
    m.qm['MEOH'].free()
    #m.K[i].free()
    if isoth=='MLL':
        #m.H[i].fix(dict_H[i])
        m.H['MEOH'].free()
        1
m.ep.free()     
#m.Ex1_cons.activate()
#m.Raff1_cons.activate()
#m.Ex2_cons.activate()
#m.Raff2_cons.activate()
#m.Recy_cons.activate()

qm_batch=dict_qm
K_batch=dict_K
H_batch=dict_H
kapp_BT=dict_kapp

R=.1
data_weight={1:1,2:1,3:1}

#m.obj=Objective(expr=sum(((m.C[di,i,j,m.t.card(-1),1]-dict_exp_conc[di,i,j])/(dict_exp_conc[di,i,j]+1e-5))**2*data_weight[di] for di in m.data for i in m.col-[1,4,5,6,7,8] for j in m.comp-['GA']))
#m.obj=Objective(expr=sum(((m.CE[di,i,j,m.t.card(-1),1]/sum(m.C[di,i,v,m.t.card(-1),1] for v in m.comp)-dict_exp_wt[di,i,j])/(dict_exp_wt[di,i,j]+1e-5))**2*data_weight[di] for di in m.data for i in m.col-[1,4,5,6,8] for j in m.comp-['GA','MEOH']))
m.obj=Objective(expr=sum(((m.CE[di,compi]-dict_exp_conc_ex[di,compi]/1000)/(dict_exp_conc_ex[di,compi]/1000))**2*data_weight[di] for di in m.data for compi in m.comp-['MEOH']))

#+R*sum(10*((m.kapp[i]-kapp_BT[i])/kapp_BT[i])**2+(5e-1*(m.H[i]-H_batch[i])/H_batch[i])**2+(1500000*(m.K[i]-K_batch[i])/K_batch[i])**2 +1000000*((m.qm[i]-qm_batch[i])/qm_batch[i])**2 for i in m.comp))
#m.obj=Objective(expr=3*sum(((m.CE[j]-dict_Ex[j]/1000)/(dict_Ex[j]/1000))**2 for j in m.comp)+10*sum(((m.CR[j]-dict_Raff[j]/1000)/(dict_Raff[j]/1000))**2 for j in m.comp - [3]))#+R*sum(10*((m.kapp[i]-kapp_BT[i])/kapp_BT[i])**2+(5e-1*(m.H[i]-H_batch[i])/H_batch[i])**2+(150*(m.K[i]-K_batch[i])/K_batch[i])**2 for i in m.comp))
#m.obj=Objective(expr=sum(((m.kapp[i]-kapp_BT[i])/kapp_BT[i])**2+((m.H[i]-H_batch[i])/H_batch[i])**2+1000*((m.K[i]-K_batch[i])/K_batch[i])**2 +10000*((m.qm[i]-qm_batch[i])/qm_batch[i])**2 for i in m.comp))
#m.obj.deactivate()
# -------------------------------------------------------------------
# Solver options
# -------------------------------------------------------------------
#%%
#### Create the ipopt_sens solver plugin using the ASL interface
solver = 'ipopt_sens'
solver_io = 'nl'
stream_solver = True    # True prints solver output to screen
keepfiles =     False    # True prints intermediate file names (.nl,.sol,...)
opt = SolverFactory(solver)#,solver_io=solver_io)
####

opt.options['mu_init'] = 1e-5
#opt.options['bound_mult_init_method']='mu-based'
#opt.options['ma57_pivtol'] = 1e-8
opt.options['max_iter'] = 5000
#opt.options['linear_system_scaling'] = 'mc19'
#
#opt.options['linear_solver'] = 'ma97'
#opt.options['linear_solver'] = 'ma57'
opt.options['linear_solver'] = 'ma27'

#m.preprocess()
results = opt.solve(m, tee=True)
   #%%

# Set global font properties
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 250

data_plot=data_index
plot_titles={1:'(a)',2:'(b)',3:'(c)'}

if len(data_plot)==1:
    fig,axes =plt.subplots(1,len(data_plot),sharex=True)
    axes=[axes]
else:
    fig,axes =plt.subplots(1,len(data_plot),sharex=True,figsize=(20,4))

colors=['k','turquoise','purple','orange'][:len(comps)]
dict_colors={}
for i,ci in enumerate(colors):
    dict_colors[comps[i]]=ci

r=len(m.t)
t=np.zeros(r)
for i in range(r): t[i]=m.t.card(i+1)

c=len(m.x)
x=np.zeros(c*ncols)
#for i in range(c): x[i]=m.x.card(i+1)

for di,datai in enumerate(data_plot):
    
    C_num=np.zeros((ncomp,c*ncols))
    C_exp=np.zeros((ncomp,ncols))
    
    for j,compj in enumerate(comps):
        count=0
        for i in range(ncols):
            for l in range(c):
                C_num[j,count]=value(m.C[datai,i+1,compj,m.t.card(-1),m.x.card(l+1)])
                x[count]=count
                count+=1
                C_exp[j,i]=dict_exp_wt[datai,i+1,compj]
                
    for j,compj in enumerate(comps):
        axes[di].plot(x,C_num[j]/sum(C_num)*100,c=dict_colors[compj],label=compj)
        axes[di].plot(x[c-1::c],C_exp[j]*100,'o',c=dict_colors[compj])
        
    axes[di].set_xticks(np.linspace(0,count,9), [str(int(i)) for i in np.linspace(0,L*ncols,9)])
    axes[di].set_ylabel('wt%',fontweight='bold')
    axes[di].set_xlabel('SMB Length (cm)',fontweight='bold')
    
    title_string=plot_titles[datai]
    
    axes[di].set_title(title_string)
    
    plt.legend(loc='center left',fontsize=14)
    axes[di].set_ylim((0,100))
    
    s=['D','E','F','R']
    a=['->','<-','->','<-']
    nc_string=[0,nc[0],nc[0]+nc[1],nc[0]+nc[1]+nc[2]]
    
    for i in range(nsec): 
        axes[di].annotate(s[i], xy=((nc_string[i])/ncols*len(x), 0),xytext=((nc_string[i])/ncols*len(x), 20) ,horizontalalignment="center",
        arrowprops=dict(arrowstyle=a[i],lw=1),fontsize=14)

plt.show()
