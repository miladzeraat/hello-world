import arviz as az
import numpy as np
import pandas
import matplotlib 
import matplotlib.pyplot as plt
import math
import pymc3 as pm
import pandas as pd
#get_ipython().magic('matplotlib qt5')


C1=3.5
C2=4.5
k=20

def gen(C1,C2,snoise=0.5,k=20,a=10):
    landa=np.zeros(k)
    sigma=np.zeros(k)
    sigma_1=np.zeros(k)
    sigma_2=np.zeros(k)
    np.random.seed(0)
    for i in range(k):
        landa[i]=(i)*0.1+1.05
        sigma[i]=(2*C1+2*C2/landa[i])*(pow(landa[i],1)-1/(pow(landa[i],2)))
        sigma_1[i]=sigma[i]
        sigma_1[i]=sigma[i]/(1+(landa[i]-1)/a)
        sigma_2[i]=sigma_1[i]+np.random.normal(0,snoise)
    return sigma,sigma_1,sigma_2, landa, snoise
def er(S,C1,C2):
    [sigma ,sigma_1 ,sigma_2]=gen(C1,C2)
    err=np.zeros(np.shape(S))
    err=sum(abs(S-sigma))
    return err
    
[sigma ,sigma_1 ,sigma_2,landa, snoise]=gen(C1,C2)
plt.plot(landa,sigma,'b',label='Without noise')
plt.plot(landa,sigma_1, 'g*',label='Noise-added1')
plt.plot(landa,sigma_2, 'ro',label='Noise-added2')
    
plt.plot(landa,sigma)
plt.xlabel('Principal Stretch')
plt.ylabel('Stress')
plt.legend()

observation=sigma_2

with pm.Model() as milad1:
  C1=pm.Normal('C1',mu=3,sd=1)
  C2=pm.Normal('C2',mu=4,sd=1)
  C3=pm.Normal('C3',mu=0,sd=100000)
  C4=pm.Normal('C4',mu=0,sd=100000)
  C5=pm.HalfNormal('C5',sd=100)
  Y=pm.Normal('Y', mu=C3+C4*landa+(2*C1+2*C2/landa)*(pow(landa,1)-1/(pow(landa,2))),sd=(snoise**2+C5**2)**0.5, observed=observation)
  trace1=pm.sample(10000,tune=1000,chains=1,random_seed=123)
az.plot_trace(trace1)
az.summary(trace1)
