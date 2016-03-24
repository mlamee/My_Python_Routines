# Two sided Kolmogorov-Smirnov (KS) and Spearman rank correlation coefficient tests
#The significance of each test is estimated by performing bootstrap sampling simulations and constructing 100,000 random same-size samples from the initial distribution. I have assigned two-tail p-values based on the results of these simulations. 

import numpy as np
import sframe as sf
from scipy import stats
import matplotlib.pyplot as plt
from math import sqrt


# This function accepts  a Numpy arrays with N elements, and make a 1D array of M*N elements with random order. 
def randomize(x,M):
    Xdeck=np.repeat(x,M,axis=0)
    np.random.shuffle(Xdeck)
    np.random.shuffle(Xdeck)
    return Xdeck

# I generally prefer to use the SFrame to read my data. In this example, I read a polarization catalog of radio galaxies with 533 rows and 51 columns.
All=sf.SFrame.read_csv('/Users/Mehdi/Dropbox/SPASS/main/SPASS-NVSS-final.csv',skiprows=52)

# I select the exact sub-sample of sources that I am interested and choose the parameters I want to run the KS and Spearman tests on.
# num is number of simulations or bootstraped samples
num=100000
# Making the target original sample from catalog. Here I perform multiple queries to choose my objects. 
sample=All[(All['W1snr'] >= 5) & (All['W2snr'] >= 5)& (All['W3snr'] >= 2)& ((All['W2er']**2+All['W3er']**2).apply(lambda x: sqrt(x)) < 0.4)]
# Making Numpy arrays of the two quantities that going to be subject of the tests
var1=sample['Alpha'].to_numpy()
var2=sample['W1'].to_numpy()-sample['W2'].to_numpy()


# Dividing the orginal observed sample into two subsamples based on the median value of parameter var1. 
#Note that the KS test will be performed on var2. but the Spearman measure the correlation coefficient between var1 and var2
med=np.median(var1)
half1=var2[np.where(var1< med)]
half2=var2[np.where(var1>= med)]
ks_main=stats.mstats.ks_twosamp(half1,half2)
print 'KS test ', ks_main
rho_main, spvalue =stats.spearmanr(var1,var2)
print 'Spearman rank correlation ', rho_main
print 'Spearman p-value', spvalue

# Randomizing and making num=100000 same size samples
var2_random=randomize(var2,num)
print np.shape(var2_random)
var1_random=randomize(var1,num)


# Doing the bootstarp analysis to calculate the correct p-values based on the observed distributions 
ks1=np.zeros(num)-1
pval=np.zeros(num)-1
rho=np.zeros(num)-1
sp_value=np.zeros(num)-1
for i in range(num):
    j=i*len(var1)
    var11=var1_random[j:j+len(var1)]
    var22=var2_random[j:j+len(var1)]
    medi=np.median(var11)
    half1=var22[np.where(var11 < medi )]
    half2=var22[np.where(var11 >= medi)]
    res=stats.mstats.ks_twosamp(half1,half2)
    ks1[i]=res[0]
    pval[i]=res[1]
    res2=stats.spearmanr(var11,var22)
    rho[i] =res2[0]
    sp_value[i]=res2[1]
print 'Done!'

howmany=np.where(ks1 >=ks_main[0])
frac=len(howmany[0])/float(num)
print 'The simulated p-vale for the KS statistic in var2 based on var1', frac 
print len(howmany[0])

howmany2=np.where(np.absolute(rho) >=np.absolute(rho_main))
frac2=len(howmany2[0])/float(num)
print 'The simulated p-vale for the Spearman rho in var1 and var2 for the simulation is', frac2 
print len(howmany2[0])

# ploting the distribution of the KS statistics based on our 100000 simuations
# Also overploting the KS statistic of the original observed sample
fig=plt.hist(ks1,bins=50,label='1e5 simulated samples')
plt.title('True p-value = '+str(frac))
plt.xlabel(r'KS statistic on var2 based on var1')
plt.plot([ks_main[0],ks_main[0]],[0,8000],color='red',label=(r'Observed sample'))
plt.legend(loc='upper right',bbox_to_anchor=[1,1],ncol=1,shadow=True,fancybox=True)
plt.savefig('TrueKS_pvalue.png')

# ploting the distribution of the Spearman rank correlation coefficients based on our 100000 simuations
# Also overploting the coefficient of the original observed sample
fig=plt.hist(rho,bins=50,label='1e5 simulated samples')
plt.title('True p-value = '+str(frac2))
plt.xlabel(r'Spearman $\rho$ on var1 and var2')
plt.plot([rho_main,rho_main],[0,8000],color='red',label=(r'Observed sample'))
plt.plot([-1*rho_main,-1*rho_main],[0,8000],color='red')
plt.legend(loc='upper right',bbox_to_anchor=[1,1],ncol=1,shadow=True,fancybox=True)
plt.savefig('TrueSpearman_pvalue.png')
