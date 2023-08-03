#!/usr/bin/env python
# coding: utf-8

# #### m = Number of payments per period (e.g., m=2 for semiannually payments)
# #### t = Number of years to maturity
# #### ytm = Yield to maturity (in decimals terms)
# #### fv = The Bond’s Face Value
# #### c = Coupon rate (in decimals terms)
#     
# #### bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
#     
# #### print (bondPrice)

# In[1]:


import os
os.getcwd()


# In[2]:


import FIA
import matplotlib.pyplot as plt


# In[3]:


bond4yr = FIA.create_coupon_bond(maturity=3, face=1000, rate=10, frequency=1)


# In[4]:


price4 = bond4yr.price(10, compounding=1)
price4


# In[2]:


def bond_price(m, t, ytm, fv, c):
    bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
    return bondPrice


# In[3]:


m = 1
t = 3
ytm = 0.1
fv = 1000
c = 0.1
    
bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
    
print (bondPrice)


# In[4]:


import numpy as np
import pandas as pd


# In[7]:


# Debt security 1

m = 1
t = 3
ytm = np.array([0.1, 0.09, 0.105, 0.12])
fv = 1000
c = 0.1

s = np.array([0.0,0.0,0.0,0.0])
for i in range(t+1):
    if i != 0:
        s[i] = bond_price(m, t-i, ytm[i], fv, c) + fv * c
        print(bond_price(m, t-i, ytm[i], fv, c) + fv*c)
    else:
        s[i] = bond_price(m, t-i, ytm[i], fv, c)
        print(bond_price(m, t-i, ytm[i], fv, c))
        
print()
print(s)


# In[7]:


# Debt security 2

m = 1
t = 3
ytm = np.array([0.1, 0.09, 0.105, 0.12])
fv = 1000
c = 0.06

for i in range(t+1):
    if i != 0:
        print(bond_price(m, t-i, ytm[i], fv, c) + fv*c)
    else:
        print(bond_price(m, t-i, ytm[i], fv, c))


# In[8]:


# Debt security 3

m = 1
t = 3
ytm = np.array([0.1, 0.09, 0.105, 0.12])
fv = 1000
c = 0.13

for i in range(4):
    if i != 0:
        print(bond_price(m, t-i, ytm[i], fv, c) + fv*c)
    else:
        print(bond_price(m, t-i, ytm[i], fv, c))


# In[9]:


# Debt security 4

m = 1
t = 3
ytm = np.array([0.1, 0.09, 0.105, 0.12])
fv = 1000
c = 0.0

for i in range(4):
    if i != 0:
        print(bond_price(m, t-i, ytm[i], fv, c) + fv * c)
    else:
        print(bond_price(m, t-i, ytm[i], fv, c))


# In[10]:


df = pd.DataFrame([[1, 3, np.array([0.1, 0.09, 0.105, 0.12]), 1000, 0.1]], columns=['m', 't', 'ytm', 'fv', 'c'])
new_row = pd.DataFrame([[1, 3, np.array([0.1, 0.09, 0.105, 0.12]), 1000, 0.06]], columns=['m', 't', 'ytm', 'fv', 'c'])
df = df.append(new_row, ignore_index=True)
new_row = pd.DataFrame([[1, 3, np.array([0.1, 0.09, 0.105, 0.12]), 1000, 0.13]], columns=['m', 't', 'ytm', 'fv', 'c'])
df = df.append(new_row, ignore_index=True)
new_row = pd.DataFrame([[1, 3, np.array([0.1, 0.09, 0.105, 0.12]), 1000, 0.0]], columns=['m', 't', 'ytm', 'fv', 'c'])
df = df.append(new_row, ignore_index=True)


# In[11]:


def bond_price_row(row):
    m = row['m']
    t = row['t']
    ytm = row['ytm']
    fv = row['fv']
    c = row['c']

    s = np.array([0,0,0,0])
    for i in range(t+1):
        if i != 0:
            s[i] = bond_price(m, t-i, ytm[i], fv, c) + fv * c
        else:
            s[i] = bond_price(m, t-i, ytm[i], fv, c)
    return s


# In[12]:


def bond_price_row1(row):
    return  ((row['fv']*row['c']/row['m']*(1-(1+row['ytm']/row['m'])**(-row['m']*row['t'])))/(row['ytm']/row['m']) + row['fv']*(1+(row['ytm']/row['m']))**(-row['m']*row['t']))
           # ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)


# In[13]:


df['bond_price'] = df.apply(bond_price_row, axis=1)


# In[14]:


df


# In[15]:


def bond_price(m, t, ytm, fv, c, erp):
    if erp == 0:
        bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
    else:
        bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + erp*(1+(ytm/m))**(-m*t)
    return bondPrice


# In[16]:


# Debt security 4 improved method

m = 1
t = 3
ytm = np.array([0.1, 0.09, 0.105, 0.12])
fv = 1000
erp = 0
c = 0.0

inf = 0
cpi = 0

for i in range(4):
    if i != 0:
        print(bond_price(m, t-i, ytm[i], fv, c, erp) + fv * c)
    else:
        print(bond_price(m, t-i, ytm[i], fv, c, erp))


# In[17]:


# Debt security 5

m = 1
t = 3
ytm = np.array([0.1, 0.09, 0.105, 0.12])
fv = 1000
c = 0.05

inf = np.array([0.0524, 0.045, 0.0651, 0.0512])
cpi = np.array([1.0, 1.045, 1.113, 1.17])

erp = np.array([0.0,0.0,0.0,0.0])
for x in range(t+1):
    erp[x] = fv * cpi[x] * pow((1 + inf[x]),t-x)

s = np.array([0.0,0.0,0.0,0.0])
for i in range(t+1):
    if i != 0:
        s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i]) + fv * c
    else:
        s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i])
        
print(s)


# In[18]:


erp

