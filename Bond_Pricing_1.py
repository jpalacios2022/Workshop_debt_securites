#!/usr/bin/env python
# coding: utf-8

# In[298]:


import numpy as np 

c = np.array([0, 100, 100, 1100])
t = np.arange(0, 4)
r_serie = np.array([ 0.1, 0.09, 0.105, 0.12])

s = 's'

if(s == 's'):
    r = 0.1
    d = (1. / np.power((1 + r), t))
else:
    r = r_serie
    for i in range(len(t)):
        d = (1. / np.power((1 + r[i]), t))

B = np.sum(d * c)


# In[299]:


d


# In[292]:


t


# In[297]:


B


# In[235]:


r


# In[285]:


# Code 4.1: Bond pricing calculation with discrete, annual compoudning
def bonds_price_discrete(times, cashflows, r):
    p = 0
    ps = []

    if type(r) == float:
        for i in range(len(times)):
            p += cashflows[i] / np.power((1 + r), times[i])
            ps = cashflows[i] / np.power((1 + r), times[i])
            print(ps)
    else:
        for i in range(len(times)):
            p += cashflows[i] / np.power((1 + r[i]), times[i])
            ps = cashflows[i] / np.power((1 + r[i]), times[i])
            print(ps)
        
    return p


# In[286]:


print('bonds price = {:.3f}'.format(bonds_price_discrete(t, c, r)))


# m = Number of payments per period (e.g., m=2 for semiannually payments)
# t = Number of years to maturity
# ytm = Yield to maturity (in decimals terms)
# fv = The Bond’s Face Value
# c = Coupon rate (in decimals terms)
#     
# bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
#     
# print (bondPrice)

# In[331]:


m = 1
t = 3
ytm = 0.1
fv = 1000
c = 0.1
    
bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
    
print (bondPrice)


# In[318]:


def bond_price(m, t, ytm, fv, c):
    bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
    return bondPrice


# In[333]:


import numpy as np 

m = 1
t = 3
ytm = np.array([0.1, 0.09, 0.105, 0.12])
fv = 1000
c = 0.1

for i in range(4):
    print(bond_price(m, t-i, ytm[i], fv, c))
    


# In[ ]:




