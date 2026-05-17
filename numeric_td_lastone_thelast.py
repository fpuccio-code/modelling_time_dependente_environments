import matplotlib.pyplot as plt
import math as mth
import numpy as np
import scipy
from scipy.signal import fftconvolve

#shape-rate parameters
def prms(mean,cv):
    a=1/cv 
    b=1/(mean*cv) 
    return [a,b]

#unstructured cv=1
def unstructured(d,b,x):
    num=d-d*np.exp(x*(b-d))
    den=d-b*np.exp(x*(b-d))
    return num/den

#renewal age-structured steady
def surv_as_steady(mu_0,cv_0,mu_2,cv_2):
    alpha0=prms(mu_0,cv_0)[0]
    beta0=prms(mu_0,cv_0)[1]
    alpha2=prms(mu_2,cv_2)[0]
    beta2=prms(mu_2,cv_2)[1]
    dx=0.001
    x=np.arange(0,5,dx)
    dd=scipy.stats.gamma.pdf(x,a=alpha0, scale= 1/beta0)*(1-scipy.stats.gamma.cdf(x,a=alpha2, scale= 1/beta2))#np.array([scipy.stats.gamma.pdf(j,a=alpha0, scale= 1/beta0)*(1-scipy.stats.gamma.cdf(j,a=alpha2, scale= 1/beta2))*dx for j in age_domain])
    gg=scipy.stats.gamma.pdf(x,a=alpha2, scale= 1/beta2)*(1-scipy.stats.gamma.cdf(x,a=alpha0, scale= 1/beta0))#np.array([scipy.stats.gamma.pdf(j,a=alpha2, scale= 1/beta2)*(1-scipy.stats.gamma.cdf(j,a=alpha0, scale= 1/beta0))*dx for j in age_domain])
    tilded=np.sum(dd)*dx
    tildeg=np.sum(gg)*dx
    if tilded>tildeg:
        return 1
    else:
        return tilded/tildeg  

#rate array function
def rate_function_np(x,mu,cv):
    alpha=prms(mu,cv)[0]
    beta=prms(mu,cv)[1]
    num=scipy.stats.gamma.pdf(x,a=alpha, scale= 1/beta)
    den=scipy.stats.gamma.sf(x,a=alpha, scale= 1/beta)#1-scipy.stats.gamma.cdf(x,a=alpha, scale= 1/beta)
    rate_ratio=num/den
    new_rate = np.nan_to_num(rate_ratio, nan=beta)
    new_rate[new_rate>beta]=beta  
    return new_rate

#extinction time distribution (alpha-dependent)
def intime_extinction_k(mu_0,cv_0,mu_2,cv_2,x,alpha):
    #mu_0, cv_0=0.3, 1
    #mu_2, cv_2=0.1, 1
    #dx=0.001
    dx=x[1]-x[0]
    array_rate0=rate_function_np(x,mu_0,cv_0)
    array_rate2=rate_function_np(x,mu_2,cv_2)

    cum_0=np.cumsum(array_rate0)*dx
    cum_2=np.cumsum(array_rate2)*dx

    cum_sum=cum_0+cum_2
    i_idx, j_idx =np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[0]), indexing='ij')
    ij = i_idx + j_idx
    ij_clipped = np.minimum(ij, x.shape[0]-1)
    mask = ij > (x.shape[0] - 1)
    term1 = np.exp(-(cum_sum[j_idx]))  # case i+j > N-1
    term2 = np.exp(-(cum_sum[ij_clipped]) + cum_sum[i_idx])  # case i+j <= N-1
    pi = np.where(mask, term1, term2)
    pig0 = np.where(ij > x.shape[0] - 1, array_rate0[-1], array_rate0[np.minimum(ij, x.shape[0] - 1)]) * pi
    pig2 = np.where(ij > x.shape[0] - 1, array_rate2[-1], array_rate2[np.minimum(ij, x.shape[0] - 1)]) * pi

    Z0=np.zeros((x.shape[0],x.shape[0]))
    Ziter=np.zeros((x.shape[0],x.shape[0]))

    l0=np.cumsum(pig0,axis=1)*dx
    iteration=0

    while iteration<10:
        iteration=iteration+1
        if alpha==0:
            Z_sq = Z0[0]**2
            l2 = np.array([fftconvolve(Z_sq, pig2[i], mode='full')[:x.shape[0]]for i in range(x.shape[0])]) * dx
            Ziter = l0 + l2
            if Ziter[0][x.shape[0]-1]-Z0[0][x.shape[0]-1]<0.00001:
                iteration=1000
        else:
            print(iteration)
            Z_sq = Z0**2
            for i in range(x.shape[0]):
                for tj in range(x.shape[0]):
                    u = np.arange(tj)
                    Ziter[i, tj] = np.sum(pig2[i, u] * Z_sq[np.minimum(i+u, x.shape[0]-1), tj-u]) * dx
            Ziter=Ziter+l0
            print(Ziter[0][x.shape[0]-1])
            if Ziter[0][x.shape[0]-1]-Z0[0][x.shape[0]-1]<0.00001:
                iteration=1000
        Z0=Ziter
    return Z0

#initial age-distribution for extinction time distribution
def intime_extinction_distribution(mu_0,cv_0,mu_2,cv_2,x,alpha,distr,N0,mu_distr,cv_distr,Z0):
#    Z0=intime_extinction_k(mu_0,cv_0,mu_2,cv_2,x,alpha)
    dx=x[1]-x[0]
    if distr=="GAMMA":
        alpha=prms(mu_distr,cv_distr)[0]
        beta=prms(mu_distr,cv_distr)[1]
        pdfgamma=scipy.stats.gamma.pdf(x,a=alpha, scale= 1/beta)
        pdfgamma=pdfgamma/(np.sum(pdfgamma))
       # pdfgamma=np.zeros(x.shape[0])
       # pdfgamma[0]=1#np.zeros(x.shape[0])
        Z_distr=np.exp(N0*np.transpose(np.log(Z0))@pdfgamma)
        return Z_distr
    elif distr=="GAUSS":
        beta=prms(mu_distr,cv_distr)[1]
        pdfgauss=scipy.stats.norm.pdf(x, loc=mu_distr, scale=1/beta) 
        pdfgauss=pdfgauss/np.sum(pdfgauss)#scipy.stats.norm.pdf(x, loc=mu_distr, scale=1/beta) 
        
        Z_distr=np.exp(N0*np.transpose(np.log(Z0))@pdfgauss)
        return Z_distr
    elif distr=="LOG":
        pdflog = lognorm.pdf(x, s=cv_distr, scale=np.exp(mu_distr))
        #pdfgauss=scipy.stats.norm.pdf(x, loc=mu_distr, scale=1/beta) 
        pdflog=pdflog/np.sum(pdflog)
        Z_distr=np.exp(N0*np.transpose(np.log(Z0))@pdfgauss)
        return Z_distr

#extinction time distribution (alpha-dependent)
def steady_extinction_k(mu_0,cv_0,mu_2,cv_2,x,alpha,mode):
    dx=x[1]-x[0]
    array_rate0=rate_function_np(x,mu_0,cv_0)
    array_rate2=rate_function_np(x,mu_2,cv_2)

    cum_0=np.cumsum(array_rate0)*dx
    cum_2=np.cumsum(array_rate2)*dx

    cum_sum=cum_0+cum_2
    i_idx, j_idx =np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[0]), indexing='ij')
    ij = i_idx + j_idx
    ij_clipped = np.minimum(ij, x.shape[0]-1)
    mask = ij > (x.shape[0] - 1)
    term1 = np.exp(-(cum_sum[j_idx]))  # case i+j > N-1
    term2 = np.exp(-(cum_sum[ij_clipped]) + cum_sum[i_idx])  # case i+j <= N-1
    pi = np.where(mask, term1, term2)
    pig0 = np.where(ij > x.shape[0] - 1, array_rate0[-1], array_rate0[np.minimum(ij, x.shape[0] - 1)]) * pi
    pig2 = np.where(ij > x.shape[0] - 1, array_rate2[-1], array_rate2[np.minimum(ij, x.shape[0] - 1)]) * pi

    tildeg2x=pig2.sum(axis=1)*dx
    tildeg0x=pig0.sum(axis=1)*dx

    if mode=="fast":#faster computation modality
        if alpha==0:
            p_x=np.zeros(x.shape[0])
            for _ in range(100):
                l=tildeg0x[0]+tildeg2x[0]*(p_x[0]**2)
                p_x[0]=l
            for j in range(x.shape[0]):
                p_x[j]=tildeg0x[j]+tildeg2x[j]*(p_x[0]**2)
            return p_x
        else:
            print("we have no fast method per alpha not 0")
    elif mode=="slow":
        #exitnction asymptotic (array Z,Z0)
        Z0=np.zeros(x.shape[0])
        ZX=np.zeros(x.shape[0])
        Ziter=np.zeros(x.shape[0])

        iteration=0
        if alpha==0:
            while iteration<10:
                iteration=iteration+1
                Z_sq = Z0**2
                Ziter=fftconvolve(Z_sq, pig2[0], mode='full')[:x.shape[0]]+pig0[0].sum()*dx
                Z0=Ziter
                if Z0[-1]-Ziter[-1]<0.0001:
                    iteration=1000
            for i in range(x.shape[0]):
                ###here fftconvolve is set to be a scalar [x.shape] npt [:x.shape]
                ZX[i]=fftconvolve(Z_sq, pig2[i], mode='full')[x.shape[0]]+pig0[i].sum()*dx
            return ZX
        
        else:#alpha not zero
            l0=np.cumsum(pig0,axis=1)*dx
            Z0=np.zeros((x.shape[0],x.shape[0]))
            Ziter=np.zeros((x.shape[0],x.shape[0]))
            while iteration<10:
                iteration=iteration+1
                print(iteration)
                Z_sq = Ziter**2
                for i in range(x.shape[0]):
                    for tj in range(x.shape[0]):
                        u = np.arange(tj)
                        u2 = (alpha * (i + u) / dx).astype(int)
                        u2 = np.minimum(u2, x.shape[0]-1)
                        Ziter[i, tj] = np.sum( pig2[i, u] * Z_sq[u2, tj-u]) * dx
                        #u = np.arange(tj)
                        #Ziter[i, tj] = np.sum(pig2[i, u] * Z_sq[np.minimum(i+u, x.shape[0]-1), tj-u]) * dx
                Ziter=Ziter+l0
                print(Ziter[0][0])
                if Ziter[0][0]-Z0[0][0]<0.00001:
                    iteration=1000
                Z0=Ziter
            return Z0[:][x.shape[0]-1]

#####################################################################################################################
#TIME-DEPENDENT ENVIRONEMNTS
def un_extinction_timedep(T_period1,T_period2,mu_death_1,mu_death_2,mu_birth_1,mu_birth_2):
    b_di_2=prms(mu_birth_2,1)[1]
    b_di_1=prms(mu_birth_1,1)[1]
    b_de_2=prms(mu_death_2,1)[1]
    b_de_1=prms(mu_death_1,1)[1]
 
    def G(h, t, b, d):
        exp_term = mth.exp((b - d) * t)
        numerator = d+d* (h - 1)* exp_term - b*h
        denominator = d+b* (h - 1)*exp_term - b*h
        return numerator / denominator
    p=0
    for j in range(50):
        #print(p)
        p= G(G(p, T_period2, b_di_2,  b_de_2), T_period1, b_di_1,  b_de_1)
    return p


####AGE-STRUCTURED TIME DEPENDENT (IT ACTUALLY WORKS)
def as_pextinction_timedep_changeT1_T2_OG(dx,T1_period,T2_period,mu_k0_env1,cv_k0_env1,mu_k0_env2,cv_k0_env2,mu_k2_env1,cv_k2_env1,mu_k2_env2,cv_k2_env2):
   #time_range=np.arange(0,20,dx)
    Nconv=7
    if T1_period>T2_period:
        Nconv=int(10/T1_period)
        x=np.arange(0,Nconv*T1_period,dx)
    else:
        Nconv=int(10/T1_period)
        x=np.arange(0,Nconv*T2_period,dx)

    T1_period_ind=int(T1_period/dx)
    T2_period_ind=int(T2_period/dx)

    T1range=x[:T1_period_ind]
    T2range=x[:T2_period_ind]
    dx=x[1]-x[0]

    #environemnt 1
    array_ratek0_env1=rate_function_np(x,mu_k0_env1,cv_k0_env1)
    array_ratek2_env1=rate_function_np(x,mu_k2_env1,cv_k2_env1)
    cum_k0_env1=np.cumsum(array_ratek0_env1)*dx
    cum_k2_env1=np.cumsum(array_ratek2_env1)*dx
    cum_sum_env1=cum_k0_env1+cum_k2_env1

    #environemnt 2
    array_ratek0_env2=rate_function_np(x,mu_k0_env2,cv_k0_env2)
    array_ratek2_env2=rate_function_np(x,mu_k2_env2,cv_k2_env2)
    cum_k0_env2=np.cumsum(array_ratek0_env2)*dx
    cum_k2_env2=np.cumsum(array_ratek2_env2)*dx
    cum_sum_env2=cum_k0_env2+cum_k2_env2

    ##Lattice - Grid
    i_idx, j_idx =np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[0]), indexing='ij')
    ij = i_idx + j_idx
    ij_clipped = np.minimum(ij, x.shape[0]-1)
    mask = ij > (x.shape[0] - 1)

    ##environmnet 1 probability matrices
    term1_env1 = np.exp(-(cum_sum_env1[j_idx]))  # case i+j > N-1
    term2_env1 = np.exp(-(cum_sum_env1[ij_clipped]) + cum_sum_env1[i_idx])  # case i+j <= N-1
    pi_env1 = np.where(mask, term1_env1, term2_env1)
    pig_k0_env1 = np.where(ij > x.shape[0] - 1, array_ratek0_env1[-1], array_ratek0_env1[np.minimum(ij, x.shape[0] - 1)]) * pi_env1
    pig_k2_env1 = np.where(ij > x.shape[0] - 1, array_ratek2_env1[-1], array_ratek2_env1[np.minimum(ij, x.shape[0] - 1)]) * pi_env1

    ##environmnet 2 probability matrices
    term1_env2 = np.exp(-(cum_sum_env2[j_idx]))  # case i+j > N-1
    term2_env2 = np.exp(-(cum_sum_env2[ij_clipped]) + cum_sum_env2[i_idx])  # case i+j <= N-1
    pi_env2 = np.where(mask, term1_env2, term2_env2)
    pig_k0_env2 = np.where(ij > x.shape[0] - 1, array_ratek0_env2[-1], array_ratek0_env2[np.minimum(ij, x.shape[0] - 1)]) * pi_env2
    pig_k2_env2 = np.where(ij > x.shape[0] - 1, array_ratek2_env2[-1], array_ratek2_env2[np.minimum(ij, x.shape[0] - 1)]) * pi_env2

    #declare time arrays F0=Z1(0,t) and G0=Z2(0,t)
    G0=np.zeros(T2range.shape[0]) 
    F0=np.zeros(T1range.shape[0])

    #declare age-arrays FT=Z1(X,T) and GT=Z2(X,T)
    GT=np.zeros(x.shape[0])
    FT=np.zeros(x.shape[0])

    #\int Pi_d(x,u)du (effective death prob.)
    l0_env1=np.cumsum(pig_k0_env1,axis=1)*dx
    l0_env2=np.cumsum(pig_k0_env2,axis=1)*dx

    iteration=0
    N_iter_max=50

    while iteration<N_iter_max:
        test_f0=F0[-1]
        test_g0=G0[-1]
        print("iter ",iteration)
        #print(F0[-1])
        iteration=iteration+1   
    
        #compute F(0,t) (solved over the time range 0-T)
        F_sq = F0**2
        
        ########check last change
        l2_env1 = np.array(fftconvolve(F_sq, pig_k2_env1[0,:T1range.shape[0]].ravel(), mode='full')[:T1range.shape[0]])*dx   
        #l2_env1 = np.convolve(F_sq, pig_k2_env1[0,:T1range.shape[0]].ravel(), mode='full')[:T1range.shape[0]] * dx
        
        F0 = l0_env1[0,:T1range.shape[0]] +  l2_env1 + pi_env1[0,:T1range.shape[0]]*GT[:T1range.shape[0]]
        np.clip(F0, 0.0, 1.0, out=F0)

        #compute F(X,T)
        FT[0]=F0[-1]         
        for i in range(x.shape[0]):
        # # here fftconvolve is set to be a scalar [x.shape] npt [:x.shape]
            if i!=0:
                FT[i]= l0_env1[i,T1_period_ind] +  fftconvolve(F_sq[0:T1_period_ind], pig_k2_env1[i,0:T1_period_ind].ravel(), mode='full')[T1_period_ind]*dx+pi_env1[i,T1_period_ind]*GT[min(T1range.shape[0]+i,x.shape[0]-1)] 
                #FT[i]=l0_env1[i,T1_period_ind]  + np.convolve(F_sq, pig_k2_env1[i,:T1range.shape[0]].ravel(), mode='full')[T1_period_ind] * dx +pi_env1[i,T1_period_ind]*GT[min(T1range.shape[0]+i,x.shape[0]-1)] 
        np.clip(FT, 0.0, 1.0, out=FT)

        #compute G(0,t)
        G_sq = G0**2
        l2_env2 = np.array(fftconvolve(G_sq, pig_k2_env2[0,:T2range.shape[0]].ravel(), mode='full')[:T2range.shape[0]] ) * dx   
        #l2_env2 = np.convolve(F_sq, pig_k2_env2[0,:T1range.shape[0]].ravel(), mode='full')[:T2range.shape[0]] * dx
        G0 = l0_env2[0,:T2range.shape[0]] +  l2_env2 + pi_env2[0,:T2range.shape[0]]*FT[:T2range.shape[0]]
        np.clip(G0, 0.0, 1.0, out=G0)

        GT[0]=G0[-1]
         # # compute G(X,T)
        for i in range(x.shape[0]):
        # # here fftconvolve is set to be a scalar [x.shape] npt [:x.shape]
            if i!=0:
                GT[i]= l0_env2[i,T2_period_ind] + fftconvolve(G_sq[0:T2_period_ind], pig_k2_env2[i,0:T2_period_ind].ravel(), mode='full')[T2_period_ind]*dx + pi_env2[i,T2_period_ind]*FT[min(T2range.shape[0]+i,x.shape[0]-1)] 
                #GT[i]=l0_env2[i,T2_period_ind]  + np.convolve(G_sq, pig_k2_env2[i,:T2range.shape[0]].ravel(), mode='full')[T2_period_ind] * dx +pi_env2[i,T2_period_ind]*FT[min(T2range.shape[0]+i,x.shape[0]-1)]         
        np.clip(GT, 0.0, 1.0, out=GT)

        if G0[-1]-test_g0<0.001 and F0[-1]-test_f0<0.001 and iteration>5:
            iteration=N_iter_max+1   
    return F0[-1]


#####AGE STRUCTURED TIME DEPENDENT ALPHA:
def as_pext_ALPHA_TEST(alpha,dx,T1_period,T2_period,mu_k0_env1,cv_k0_env1,mu_k0_env2,cv_k0_env2,mu_k2_env1,cv_k2_env1,mu_k2_env2,cv_k2_env2):
    Nconv=7
    if T1_period>T2_period:
       # Nconv=int(10/T1_period)
        Nconv = max(2, int(5/ T1_period))
        x=np.arange(0,Nconv*T1_period,dx)
    else:
        #Nconv=int(10/T1_period)
        Nconv = max(2, int(5 / T1_period))
        x=np.arange(0,Nconv*T2_period,dx)
   # print(x)
    T1_period_ind=int(T1_period/dx)
    T2_period_ind=int(T2_period/dx)

    T1range=x[:T1_period_ind]
    T2range=x[:T2_period_ind]

    dx=x[1]-x[0]

    #environemnt 1
    array_ratek0_env1=rate_function_np(x,mu_k0_env1,cv_k0_env1)
    array_ratek2_env1=rate_function_np(x,mu_k2_env1,cv_k2_env1)
    cum_k0_env1=np.cumsum(array_ratek0_env1)*dx
    cum_k2_env1=np.cumsum(array_ratek2_env1)*dx
    cum_sum_env1=cum_k0_env1+cum_k2_env1

    #environemnt 2
    array_ratek0_env2=rate_function_np(x,mu_k0_env2,cv_k0_env2)
    array_ratek2_env2=rate_function_np(x,mu_k2_env2,cv_k2_env2)
    cum_k0_env2=np.cumsum(array_ratek0_env2)*dx
    cum_k2_env2=np.cumsum(array_ratek2_env2)*dx
    cum_sum_env2=cum_k0_env2+cum_k2_env2

    ##Lattice - Grid
    i_idx, j_idx =np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[0]), indexing='ij')
    ij = i_idx + j_idx
    ij_clipped = np.minimum(ij, x.shape[0]-1)
    mask = ij > (x.shape[0] - 1)

    ##environmnet 1 probability matrices
    term1_env1 = np.exp(-(cum_sum_env1[j_idx]))  # case i+j > N-1
    term2_env1 = np.exp(-(cum_sum_env1[ij_clipped]) + cum_sum_env1[i_idx])  # case i+j <= N-1
    pi_env1 = np.where(mask, term1_env1, term2_env1)
    pig_k0_env1 = np.where(ij > x.shape[0] - 1, array_ratek0_env1[-1], array_ratek0_env1[np.minimum(ij, x.shape[0] - 1)]) * pi_env1
    pig_k2_env1 = np.where(ij > x.shape[0] - 1, array_ratek2_env1[-1], array_ratek2_env1[np.minimum(ij, x.shape[0] - 1)]) * pi_env1

    ##environmnet 2 probability matrices
    term1_env2 = np.exp(-(cum_sum_env2[j_idx]))  # case i+j > N-1
    term2_env2 = np.exp(-(cum_sum_env2[ij_clipped]) + cum_sum_env2[i_idx])  # case i+j <= N-1
    pi_env2 = np.where(mask, term1_env2, term2_env2)
    pig_k0_env2 = np.where(ij > x.shape[0] - 1, array_ratek0_env2[-1], array_ratek0_env2[np.minimum(ij, x.shape[0] - 1)]) * pi_env2
    pig_k2_env2 = np.where(ij > x.shape[0] - 1, array_ratek2_env2[-1], array_ratek2_env2[np.minimum(ij, x.shape[0] - 1)]) * pi_env2

    #declare time arrays F0=Z1(0,t) and G0=Z2(0,t)
    Fmatrix=np.zeros((x.shape[0],T1range.shape[0]))
    Gmatrix=np.zeros((x.shape[0],T2range.shape[0])) 
    Fmatrix2=np.zeros((x.shape[0],T1range.shape[0]))
    Gmatrix2=np.zeros((x.shape[0],T2range.shape[0]))

    #\int Pi_d(x,u)du (effective death prob.)
    l0_env1=np.cumsum(pig_k0_env1,axis=1)*dx
    l0_env2=np.cumsum(pig_k0_env2,axis=1)*dx

    iteration=0
    N_iter_max=10

    while iteration<N_iter_max:
        test_F=Fmatrix[0,-1]
        test_G=Gmatrix[0,-1]

        print("iter ",iteration)
        iteration=iteration+1
        

        T1 = T1range.shape[0]
        X = x.shape[0]

        for i in range(X):          
            for j in range(T1):
                if j > 0:
                    k = np.arange(j)
                    idx1 = j - k
                    idx2 = np.minimum(T1-1, j + i - k)
                    u = pig_k2_env1[i, idx1] * Fmatrix[alpha * idx2, k] * Fmatrix[0, k]
                    s = np.sum(u)
            else:
                s = 0.0

            Fmatrix2[i,j] = (l0_env1[i,j]+ s * dx+ pi_env1[i,j] * Gmatrix[min(i+j, X-1), min(T1, T2range.shape[0])-1])
        
        T2 = T2range.shape[0]
        T1 = T1range.shape[0]
        X = x.shape[0]

        for i in range(X):                      
            for j in range(T2):
                if j > 0:
                    k = np.arange(j)
                    idx1 = j - k
                    idx2 = np.minimum(T2 - 1, j + i - k)
                    u = pig_k2_env2[i, idx1] * Gmatrix[alpha * idx2, k] * Gmatrix[0, k]
                    s = np.sum(u)
                else:
                    s = 0.0
                Gmatrix2[i, j] = (
                pi_env2[i, j] * Fmatrix[min(i + j, X - 1), min(T1, T2) - 1]+ l0_env2[i, j]+ s * dx)
        
        # ###ROLL OVER MATRIX F (environemnt 1)
        # for i in range(x.shape[0]):          
        #     for j in range((T1range.shape[0])):
        #         #at current state alha=1 or 0
        #         u=[pig_k2_env1[i,j-k]*Fmatrix[alpha*min(T1range.shape[0]-1,j+i-k),k]*Fmatrix[0,k]  for k in range(j)]
        #         Fmatrix2[i,j]= l0_env1[i,j] + sum(u)*dx+ pi_env1[i,j]*Gmatrix[min(i+j,x.shape[0]-1),min(T1range.shape[0],T2range.shape[0])-1] 

        # ###ROLL OVER MATRIX G (environemnt 2)         
        # for i in range(x.shape[0]):                      
        #     for j in range((T2range.shape[0])):
        #         #at current state alha=1 or 0
                
        #         u=[pig_k2_env2[i,j-k]*Gmatrix[alpha*min(T2range.shape[0]-1,j+i-k),k]* Gmatrix[0,k]  for k in range(j)]
              
        #         Gmatrix2[i,j]=pi_env2[i,j]*Fmatrix[min(i+j,x.shape[0]-1),min(T1range.shape[0],T2range.shape[0])-1] + l0_env2[i,j] + sum(u)*dx

        if Gmatrix[0,-1]-test_G<0.001 and Fmatrix[0,-1]-test_F<0.001 and iteration>5:
            iteration=N_iter_max+1   
       
        Fmatrix=Fmatrix2
        Gmatrix=Gmatrix2
    return Fmatrix[0,-1]

    
#####################################################################################################
##evaluate grwoth rate ## NOT SURE HOW TO COMPUTE FOR NOT RENEWAL
def growth_rate_alpha0(alpha,mu0,cv0,mu2,cv2):
    dx=0.01
    xrange=np.arange(0,5,dx)
    
    def fun(mu0,cv0,mu2,cv2,x):
        cdf0=(1-scipy.stats.gamma.cdf(x,a=prms(mu0,cv0)[0], scale= 1/prms(mu0,cv0)[1]))
        pdf2=scipy.stats.gamma.pdf(x,a=prms(mu2,cv2)[0], scale= 1/prms(mu2,cv2)[1])
        return pdf2*cdf0    

    test=[]
    dl=0.1
    range_growthrate=np.arange(-5,5,dl)
    for l in range_growthrate:
        print(l)
        integral=np.array([2*mth.exp(-l*j)*fun(mu0,cv0,mu2,cv_2,j)*dx for j in xrange])
        test.append(abs(integral.sum()-1))
    
    ind_min=test.index(min(test))
    return range_growthrate[ind_min]


# dx=0.01
# Tperiodo=2
# Tperiodo1=Tperiodo
# Tperiodo2=Tperiodo
# mug1=1.5
# mud1=2
# mug2=1.5
# mud2=10
# cv_d_env1=0.1
# cv_g_env1=0.1
# cv_d_env2=0.1
# cv_g_env2=0.1
# #mu_k0,cv_k0,mu_k2,cv_k2=1,1,1,1

# #y,domain=as_pext_ALPHA_TEST(0,0.01,Tperiodo1,Tperiodo2,mud1,cv_d_env1,mud2,cv_d_env2,mug1,cv_g_env1,mug2,cv_g_env2)
# newalpha=as_pext_ALPHA_TEST(1,dx,Tperiodo1,Tperiodo2,mud1,cv_d_env1,mud2,cv_d_env2,mug1,cv_g_env1,mug2,cv_g_env2)
# sure=as_pextinction_timedep_changeT1_T2_OG(dx,Tperiodo1,Tperiodo2,mud1,cv_d_env1,mud2,cv_d_env2,mug1,cv_g_env1,mug2,cv_g_env2)
# print("newalpha",newalpha)
# print("sure",sure)

# #plt.plot(domain,y)
# plt.show()

# exit()


dx=0.01
rangecv=[1,0.5,0.1,0.03]
numerics={str(i):[] for i in rangecv}

numerics_alpha={str(i):[] for i in rangecv}

ranging=np.arange(0.5,2,0.05)
for coeff in rangecv: 
    for k in ranging:
        print("TPeriod",k)
        Tperiodo=k
        Tperiodo1=Tperiodo
        Tperiodo2=Tperiodo

        mug1=1.5
        mud1=2
        mug2=1.5
        mud2=20
        cv_d_env1=0.03
        cv_g_env1=0.03
        cv_d_env2=0.03
        cv_g_env2=0.03
        
        cv_d_env1=coeff
        cv_d_env2=coeff
        
        numerics[str(coeff)].append(1-as_pextinction_timedep_changeT1_T2_OG(dx,Tperiodo1,Tperiodo2,mud1,cv_d_env1,mud2,cv_d_env2,mug1,cv_g_env1,mug2,cv_g_env2))
        numerics_alpha[str(coeff)].append(1-as_pext_ALPHA_TEST(1,dx,Tperiodo1,Tperiodo2,mud1,cv_d_env1,mud2,cv_d_env2,mug1,cv_g_env1,mug2,cv_g_env2))

colors=["red","blue","orange","black","purple"]
#print(as_exact)
i=0
for _ in rangecv:
    i=i+1
    plt.plot(ranging,numerics[str(_)],label="CV= "+str(_),color=colors[i]) #,marker="o",)
    plt.plot(ranging,numerics_alpha[str(_)],label="CVALPHA= "+str(_),linestyle="--",color=colors[i]) #,marker="o")


plt.legend()
plt.show()
