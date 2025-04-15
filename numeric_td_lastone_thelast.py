import matplotlib.pyplot as plt
import math as mth
import numpy as np
import scipy.integrate
import os
import subprocess
from scipy.integrate import cumulative_trapezoid

def prms(mean,cv):
    a=1/cv #shape
    b=1/(mean*cv) #rate
    return [a,b]


MU_DE_ON = float(os.getenv('MU_DE_ON',  1.8))#1.3 # Mean death time
MU_DI_ON = float(os.getenv('MU_DI_ON', 2))#1.  # Mean division time
CV_DE_ON = float(os.getenv('CV_DE_ON', 0.03))#0.3  # Coefficient of variation (death time)
CV_DI_ON =float(os.getenv('CV_DI_ON', 0.03))# 0.2  # Coefficient of variation (division time)

MU_DE_OFF = float(os.getenv('MU_DE_OFF',  20))#1.3 # Mean death time
MU_DI_OFF = float(os.getenv('MU_DI_OFF', 2))#1.  # Mean division time
CV_DE_OFF = float(os.getenv('CV_DE_OFF', 0.03))#0.3  # Coefficient of variation (death time)
CV_DI_OFF =float(os.getenv('CV_DI_OFF', 0.03))# 0.2  # Coefficient of variation (division time)

MU_GO0_ON = float(os.getenv('MU_GO0_ON',20))#0.9  # Mean dormant time
MU_GO1_ON = float(os.getenv('MU_GO1_ON', 20))#1.2  # Mean awake time
CV_GO0_ON = float(os.getenv('CV_GO0_ON', 0.01))#0.2  # Coefficient of variation (dormant time)
CV_GO1_ON = float(os.getenv('CV_GO1_ON', 0.01))#0.2  # Coefficient of variation (awake time)
MU_GO0_OFF = float(os.getenv('MU_GO0_OFF', 20))#0.9  # Mean dormant time
MU_GO1_OFF = float(os.getenv('MU_GO1_OFF', 20))#1.2  # Mean awake time
CV_GO0_OFF = float(os.getenv('CV_GO0_OFF', 0.01))#0.2  # Coefficient of variation (dormant time)
CV_GO1_OFF = float(os.getenv('CV_GO1_OFF', 0.01))#0.2

ALPHA=float(os.getenv('ALPHA', 0.))
BETA=float(os.getenv('BETA', 0.))

b_death_on= prms(MU_DE_ON,CV_DE_ON)[1]
a_death_on=prms(MU_DE_ON,CV_DE_ON)[0]

b_div_on = prms(MU_DI_ON,CV_DI_ON)[1]
a_div_on=prms(MU_DI_ON,CV_DI_ON)[0]

b_go0_on = prms(MU_GO0_ON,CV_GO0_ON)[1]
a_go0_on=prms(MU_GO0_ON,CV_GO0_ON)[0]

b_go1_on = prms(MU_GO1_ON,CV_GO1_ON)[1]
a_go1_on=prms(MU_GO1_ON,CV_GO1_ON)[0]

b_death_off= prms(MU_DE_OFF,CV_DE_OFF)[1]
a_death_off=prms(MU_DE_OFF,CV_DE_OFF)[0]

b_div_off = prms(MU_DI_OFF,CV_DI_OFF)[1]
a_div_off=prms(MU_DI_OFF,CV_DI_OFF)[0]

b_go0_off = prms(MU_GO0_OFF,CV_GO0_OFF)[1]
a_go0_off=prms(MU_GO0_OFF,CV_GO0_OFF)[0]

b_go1_off = prms(MU_GO1_OFF,CV_GO1_OFF)[1]
a_go1_off=prms(MU_GO1_OFF,CV_GO1_OFF)[0]

TMAX = float(os.getenv('TMAX', 1))
XMAX = float(os.getenv('XMAX', TMAX*2))#Simulation time
MT = int(os.getenv('MT',1001))
NX = int(os.getenv('NX',1000))#Simulation time


nx=NX
mt=MT
tmax=TMAX
xmax=XMAX
xrange=np.linspace(0.000,xmax,nx)
trange=np.linspace(0.000,tmax,mt)
deltax=xrange[1]-xrange[0]
deltat=trange[1]-trange[0]

############################################################################################################################################
import numpy as np
import scipy.stats

def compute_transition_probabilities(nx, mt, trange, xrange, 
                                     a_death_on, b_death_on, a_div_on, b_div_on, a_go0_on, b_go0_on,
                                     a_death_off, b_death_off, a_div_off, b_div_off, a_go0_off, b_go0_off):
    
    # Define probability functions using NumPy operations
    def Pion(x):
        return ((1 - scipy.stats.gamma.cdf(x, a=a_death_on, scale=1/b_death_on)) *
                (1 - scipy.stats.gamma.cdf(x, a=a_div_on, scale=1/b_div_on)) *
                (1 - scipy.stats.gamma.cdf(x, a=a_go0_on, scale=1/b_go0_on)))

    def Pidon(x):
        return (scipy.stats.gamma.pdf(x, a=a_death_on, scale=1/b_death_on) *
                (1 - scipy.stats.gamma.cdf(x, a=a_div_on, scale=1/b_div_on)) *
                (1 - scipy.stats.gamma.cdf(x, a=a_go0_on, scale=1/b_go0_on)))

    def Pigon(x):
        return ((1 - scipy.stats.gamma.cdf(x, a=a_death_on, scale=1/b_death_on)) *
                scipy.stats.gamma.pdf(x, a=a_div_on, scale=1/b_div_on) *
                (1 - scipy.stats.gamma.cdf(x, a=a_go0_on, scale=1/b_go0_on)))

    def Picon(x):
        return ((1 - scipy.stats.gamma.cdf(x, a=a_death_on, scale=1/b_death_on)) *
                (1 - scipy.stats.gamma.cdf(x, a=a_div_on, scale=1/b_div_on)) *
                scipy.stats.gamma.pdf(x, a=a_go0_on, scale=1/b_go0_on))

    def Pioff(x):
        return ((1 - scipy.stats.gamma.cdf(x, a=a_death_off, scale=1/b_death_off)) *
                (1 - scipy.stats.gamma.cdf(x, a=a_div_off, scale=1/b_div_off)) *
                (1 - scipy.stats.gamma.cdf(x, a=a_go0_off, scale=1/b_go0_off)))

    def Pidoff(x):
        return (scipy.stats.gamma.pdf(x, a=a_death_off, scale=1/b_death_off) *
                (1 - scipy.stats.gamma.cdf(x, a=a_div_off, scale=1/b_div_off)) *
                (1 - scipy.stats.gamma.cdf(x, a=a_go0_off, scale=1/b_go0_off)))

    def Pigoff(x):
        return ((1 - scipy.stats.gamma.cdf(x, a=a_death_off, scale=1/b_death_off)) *
                scipy.stats.gamma.pdf(x, a=a_div_off, scale=1/b_div_off) *
                (1 - scipy.stats.gamma.cdf(x, a=a_go0_off, scale=1/b_go0_off)))

    # Create 2D arrays using broadcasting (Shape: (mt, nx))
    XJ = trange[:, np.newaxis] + xrange  # Broadcasting to create shifted values

    # Precompute denominators
    Pion_xrange = Pion(xrange)
    Pioff_xrange = Pioff(xrange)

    # Compute matrices using vectorized operations
    Piz_on  = np.nan_to_num(Pion(XJ) / Pion_xrange, nan=0)
    Pid_on  = np.nan_to_num(Pidon(XJ) / Pion_xrange, nan=0)
    Pig_on  = np.nan_to_num(Pigon(XJ) / Pion_xrange, nan=0)
    Pic_on  = np.nan_to_num(Picon(XJ) / Pion_xrange, nan=0)
    Piz_off = np.nan_to_num(Pioff(XJ) / Pioff_xrange, nan=0)
    Pid_off = np.nan_to_num(Pidoff(XJ) / Pioff_xrange, nan=0)
    Pig_off = np.nan_to_num(Pigoff(XJ) / Pioff_xrange, nan=0)

    # Transpose results to match expected shape (nx, mt)
    return Piz_on.T, Pid_on.T, Pig_on.T, Pic_on.T, Piz_off.T, Pid_off.T, Pig_off.T

Piz_on, Pid_on, Pig_on, Pic_on, Piz_off, Pid_off, Pig_off = compute_transition_probabilities(nx, mt, trange, xrange, 
                                     a_death_on, b_death_on, a_div_on, b_div_on, a_go0_on, b_go0_on,
                                     a_death_off, b_death_off, a_div_off, b_div_off, a_go0_off, b_go0_off)

########################################################################################################################
tilded=scipy.integrate.simpson([Pid_on[0,j] for j in range(mt)], x=trange)
tildeg=scipy.integrate.simpson([Pig_on[0,j] for j in range(mt)], x=trange)
tildec=scipy.integrate.simpson([Pic_on[0,j] for j in range(mt)], x=trange)
#print("tilded/tildeg",tilded/tildeg)


def zfun(b_death,b_div,h,t):
    num=b_death+b_death*(h-1)*mth.exp((b_div-b_death)*t)-b_div*h
    den=b_death+b_div*(h-1)*mth.exp((b_div-b_death)*t)-b_div*h
    return num*(1/den)

def unstru_td(rate_div,rate_de,T):
    p=0
    for _ in range(150):
        p=zfun(rate_de,rate_div,zfun(0,rate_div,p,T),T)
    return p

#print("exact:homo(d/g),",b_death_on/b_div_on)
#print("exact:td (same env)",unstru_td(b_div_on,b_death_on,tmax))
#print("exact astoric homo func",zfun(b_death_on,b_div_on,0,100))


def III_term(Pid, deltat):
    pid = cumulative_trapezoid(Pid, dx=deltat, axis=1, initial=0)
    pid = np.nan_to_num(pid, nan=0)
    
    return pid


def II_term(nx, mt, Pig, z, deltat):
    z2 = z[0, :] ** 2 
    Mz2 = np.zeros((mt, mt))  

    for i in range(mt):
        Mz2[i, i:mt:1] = z2[0:mt-i:1] 

    out=np.dot(Pig,Mz2)*deltat
    out2=np.nan_to_num(out,0)
    return out2
from scipy.integrate import simpson

def II_term_simpson(nx, mt, Pig, z, deltat):
    z2 = z[0, :] ** 2  # shape: (mt,)
    Mz2 = np.zeros((mt, mt))

    # Build Simpson's weights
    weights = np.ones(mt)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2

    weights *= deltat / 3  # Scale weights for Simpson's rule

    # Build a matrix like before: cumulative integration matrix
    for i in range(mt):
        Mz2[i, i:mt] = z2[i:mt] * weights[0:mt-i]

    out = np.dot(Pig, Mz2)
    out2 = np.nan_to_num(out, nan=0.0)
    return out2


# def I2_term(nx, mt, Piz, z, trange, deltax, xxrange, xmax):
#     """Vectorized computation of I2_term."""
#     X = trange[None, :] + xxrange[:, None]  # Shape (nx, mt)
#     indices = findx_vectorized(X, deltax, nx)  # Shape (nx, mt)
#     # Mask for values exceeding xmax
#     mask = X > xmax  # Boolean mask, Shape (nx, mt)
#     # Compute pi using broadcasting
#   #  pi = np.where(mask, Piz* z[-1, :], Piz * z[indices, np.arange(mt)])
#     pi = np.where(mask, Piz * z[-1, -1], Piz * z[indices, -1])
#     return pi


# def I2_term(nx, mt, Piz, z, trange, deltax, xxrange, xmax):
#     a=np.zeros((nx,mt))
#     for j in range(mt):
#         for i in range(nx):     
#             if i+j>nx-1:
#                 a[i,j]=Piz[i,j]*z[-1,-1]
#             else:
#                 a[i,j]=Piz[i,j]*z[i+j,-1]
#     return a



def I2_term(nx, mt, Piz, z, trange, deltax, xxrange, xmax):
    # Create index grids for i and j
    I, J = np.meshgrid(np.arange(nx), np.arange(mt), indexing='ij')
    K = I + J  # K is i + j

    # Condition: K > nx - 1
    mask = K > (nx - 1)

    # Initialize output
    a = np.empty((nx, mt))

    # Where K > nx - 1, use z[-1, -1]
    a[mask] = Piz[mask] * z[-1, -1]

    # Where K <= nx - 1, use z[K, -1]
    a[~mask] = Piz[~mask] * z[K[~mask], -1]

    return a

u1=Pid_off
u2=Pid_on

for _ in range(20):
   # print(u2[0,-1])
    #Zoff[]=
    u1 =  II_term_simpson(nx,mt,Pig_off,u1,deltat)+III_term(Pid_off,deltat)+I2_term(nx,mt,Piz_off,u2,trange, deltax,xrange,xmax) 
    #Zon[Zoff[]]=
    u2 =  II_term_simpson(nx,mt,Pig_on,u2,deltat)+III_term(Pid_on,deltat) +I2_term(nx,mt,Piz_on,u1,trange, deltax,xrange,xmax) 
    np.nan_to_num(u1,nan=0,posinf=0, neginf=0)
    np.nan_to_num(u2,nan=0,posinf=0, neginf=0)

def zfun(b_death,b_div,h,t):
    num=b_death+b_death*(h-1)*mth.exp((b_div-b_death)*t)-b_div*h
    den=b_death+b_div*(h-1)*mth.exp((b_div-b_death)*t)-b_div*h
    return num*(1/den)

##print(float(u1[0,-1]))
print(float(u2[0,-1]))
# arrw=[zfun(b_death_on,b_div_on,0,xj) for xj in trange]
#plt.plot(trange,u2[0,:],linewidth=3)
# plt.plot(trange,arrw,linestyle="dashed",label="theory",linewidth=3)
# plt.legend()
#plt.show()