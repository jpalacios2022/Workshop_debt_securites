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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import decimal


# In[ ]:


import os
os.getcwd()


# In[ ]:





# In[39]:


def bond_price(m, t, ytm, fv, c):
    bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
    return bondPrice


# In[40]:


m = 1
t = 3
ytm = 0.1
fv = 1000
c = 0.1
    
bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
    
print (bondPrice)


# In[41]:


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
    else:
        s[i] = bond_price(m, t-i, ytm[i], fv, c)
        
print(s)


# In[42]:


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


# In[43]:


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


# In[44]:


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


# In[45]:


# Debt nic case 4.01

fv = 158702533.58
c = 0.05
t = 20
m = 1
ytm = 0.06

cy = t*m

erp = 0

s = np.zeros(cy+1)
for i in range(cy+1):
    if i != 0:
        print(bond_price(m, cy-i, ytm, fv, c) + fv * c)
    else:
        print(bond_price(m, cy-i, ytm, fv, c))
        


# In[9]:


df = pd.DataFrame([[1, 3, np.array([0.1, 0.09, 0.105, 0.12]), 1000, 0.1]], columns=['m', 't', 'ytm', 'fv', 'c'])
new_row = pd.DataFrame([[1, 3, np.array([0.1, 0.09, 0.105, 0.12]), 1000, 0.06]], columns=['m', 't', 'ytm', 'fv', 'c'])
df = df.append(new_row, ignore_index=True)
new_row = pd.DataFrame([[1, 3, np.array([0.1, 0.09, 0.105, 0.12]), 1000, 0.13]], columns=['m', 't', 'ytm', 'fv', 'c'])
df = df.append(new_row, ignore_index=True)
new_row = pd.DataFrame([[1, 3, np.array([0.1, 0.09, 0.105, 0.12]), 1000, 0.0]], columns=['m', 't', 'ytm', 'fv', 'c'])
df = df.append(new_row, ignore_index=True)


# In[46]:


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


# In[47]:


df['bond_price'] = df.apply(bond_price_row, axis=1)


# In[48]:


df


# In[50]:


def bond_price(m, t, ytm, fv, c, erp):
    if erp == 0:
        bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
    else:
        bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + erp*(1+(ytm/m))**(-m*t)
    return bondPrice


# In[51]:


# Debt security 1 improved method

fv = 1000
c = 0.1
t = 3
m = 1
ytm = np.array([0.1, 0.09, 0.105, 0.12])

erp = 0
inf = 0
cpi = 0
er = 0
fe = 0
gp_nc = 0

for i in range(t+1):
    if i != 0:
        print(bond_price(m, t-i, ytm[i], fv, c, erp) + fv * c)
    else:
        print(bond_price(m, t-i, ytm[i], fv, c, erp))


# In[52]:


# Debt security 2 improved method

fv = 1000
c = 0.06
t = 3
m = 1
ytm = np.array([0.1, 0.09, 0.105, 0.12])

erp = 0
inf = 0
cpi = 0
er = 0
fe = 0

for i in range(t+1):
    if i != 0:
        print(bond_price(m, t-i, ytm[i], fv, c, erp) + fv * c)
    else:
        print(bond_price(m, t-i, ytm[i], fv, c, erp))


# In[53]:


# Debt security 3 improved method

fv = 1000
c = 0.13
t = 3
m = 1
ytm = np.array([0.1, 0.09, 0.105, 0.12])

erp = 0
inf = 0
cpi = 0
er = 0
fe = 0

for i in range(t+1):
    if i != 0:
        print(bond_price(m, t-i, ytm[i], fv, c, erp) + fv * c)
    else:
        print(bond_price(m, t-i, ytm[i], fv, c, erp))


# In[54]:


# Debt security 4 improved method

fv = 1000
c = 0.0
t = 3
m = 1
ytm = np.array([0.1, 0.09, 0.105, 0.12])

erp = 0
inf = 0
cpi = 0
er = 0
fe = 0
gp_nc = 0

for i in range(t+1):
    if i != 0:
        print(bond_price(m, t-i, ytm[i], fv, c, erp) + fv * c)
    else:
        print(bond_price(m, t-i, ytm[i], fv, c, erp))


# In[27]:


# Debt nic case 4.01

fv = 158702533.58
c = 0.05
t = 20
m = 2
ytm = 1

cy = t*m

erp = 0

s = np.zeros(cy+1)
for i in range(cy+1):
    if i != 0:
        s[i] = bond_price(m, t-i, ytm, fv, c, erp) + fv * c
    else:
        s[i] = bond_price(m, t-i, ytm, fv, c, erp)
        
print(s)


# In[ ]:





# In[58]:


# Debt nicaragua case 4.02

fv = 10000000.00
c = 0.08
t = 4
m = 2
ytm = 0.00584

cy = t*m

erp = 0

s = np.zeros(cy+1)
for i in range(cy+1):
    if i != 0:
        s[i] = bond_price(m, t-i, ytm, fv, c, erp) + fv * c
    else:
        s[i] = bond_price(m, t-i, ytm, fv, c, erp)
        
s1 = s.tolist()
s = s1
print(s)


# In[19]:


# Debt security 5

fv = 1000
c = 0.05
t = 3
m = 1
ytm = np.array([0.1, 0.09, 0.105, 0.12])

er = 0
fe = 0

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


# In[20]:


# Debt security 6

fv = 1000
c = 0.1
t = 3
m = 1
ytm = np.array([0.1, 0.09, 0.105, 0.12])

inf = 0
cpi = 0
gp_nc = 0

er = 0.1
fe = np.array([0.1, 0.093, 0.105, 0.11])

erp = np.array([0.0,0.0,0.0,0.0])
for x in range(t+1):
    erp[x] = fv / (er / fe[x])

s = np.array([0.0,0.0,0.0,0.0])
for i in range(t+1):
    if i != 0:
        s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i]) + fv * c
    else:
        s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i])
        
print(s)


# In[21]:


# Debt security 7

fv = 1000
c = 0.06
t = 3
m = 1
ytm = np.array([0.1, 0.09, 0.105, 0.12])

inf = 0
cpi = 0

gp_nc = 100
igp = 0.0423
egp_nc = np.array([100,105,110,109])

erp = np.array([0.0,0.0,0.0,0.0])
for x in range(t+1):
    if x == 0:
        erp[x] = fv * pow((1 + igp),t-x) 
    if x > 0:
        erp[x] = fv / gp_nc * egp_nc[x] * pow((egp_nc[x] / egp_nc[x-1]),t-x)
        
s = np.array([0.0,0.0,0.0,0.0])
for i in range(t+1):
    if i != 0:
        s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i]) + fv * c
    else:
        s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i])
        
print(s)


# In[22]:


def bond_price(m, t, ytm, fv, c, erp):
    if erp == 0:
        bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + fv*(1+(ytm/m))**(-m*t)
    else:
        bondPrice = ((fv*c/m*(1-(1+ytm/m)**(-m*t)))/(ytm/m)) + erp*(1+(ytm/m))**(-m*t)
    return bondPrice


# In[23]:


def bond_price_row(row):
    fv = row['fv']
    c = row['c']
    t = row['t']
    m = row['m']
    ytm = row['ytm']
    #parameter to decide which erp_type will be processed
    erp_type = row['erp_type']
    #parameters for expected return price cases
    #case 1 - cpi
    cpi = row['cpi']
    inf = row['inf']
    #case 2 - foreign exchange
    er = row['er']
    fe = row['fe']
    #case 3 - gold
    gp_nc = row['gp_nc']
    igp = row['igp']
    egp_nc = row['egp_nc']
        
    erp = 0
    if erp_type == 0:
        s = np.array([0.0,0.0,0.0,0.0])
        for i in range(t+1):
            if i != 0:
                s[i] = bond_price(m, t-i, ytm[i], fv, c, erp) + fv * c
            else:
                s[i] = bond_price(m, t-i, ytm[i], fv, c, erp)
        return np.round(s,decimals=2)
    
    elif erp_type == 1:
        erp = np.array([0.0,0.0,0.0,0.0])
        for x in range(t+1):
            erp[x] = fv * cpi[x] * pow((1 + inf[x]),t-x)

        s = np.array([0.0,0.0,0.0,0.0])
        for i in range(t+1):
            if i != 0:
                s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i]) + fv * c
            else:
                s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i])
        return np.round(s,decimals=2)
    
    elif erp_type == 2:
        erp = np.array([0.0,0.0,0.0,0.0])
        for x in range(t+1):
            erp[x] = fv / (er / fe[x])

        s = np.array([0.0,0.0,0.0,0.0])
        for i in range(t+1):
            if i != 0:
                s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i]) + fv * c
            else:
                s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i])
        return np.round(s,decimals=2)
        
    elif erp_type == 3:
        erp = np.array([0.0,0.0,0.0,0.0])
        for x in range(t+1):
            if x == 0:
                erp[x] = fv * pow((1 + igp),t-x) 
            if x > 0:
                erp[x] = fv / gp_nc * egp_nc[x] * pow((egp_nc[x] / egp_nc[x-1]),t-x)
 
        s = np.array([0.0,0.0,0.0,0.0])
        for i in range(t+1):
            if i != 0:
                s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i]) + fv * c
            else:
                s[i] = bond_price(m, t-i, ytm[i], fv, c, erp[i])
        return np.round(s,decimals=2)


# In[24]:


df = pd.DataFrame([[1000, 0.1, 3, 1, np.array([0.1, 0.09, 0.105, 0.12]),0,0,0,0,0,0,0,0]], 
                  columns=['fv','c','t','m', 'ytm','erp_type','cpi','inf','er','fe','gp_nc','igp', 'egp_nc'])
new_row = pd.DataFrame([[1000, 0.06, 3, 1, np.array([0.1, 0.09, 0.105, 0.12]),0,0,0,0,0,0,0,0]],
                 columns=['fv','c','t','m', 'ytm','erp_type','cpi','inf','er','fe','gp_nc','igp', 'egp_nc'])
df = df.append(new_row, ignore_index=True)
new_row = pd.DataFrame([[1000, 0.13, 3, 1, np.array([0.1, 0.09, 0.105, 0.12]),0,0,0,0,0,0,0,0]],
                 columns=['fv','c','t','m', 'ytm','erp_type','cpi','inf','er','fe','gp_nc','igp', 'egp_nc'])
df = df.append(new_row, ignore_index=True)
new_row = pd.DataFrame([[1000, 0.0, 3, 1, np.array([0.1, 0.09, 0.105, 0.12]),0,0,0,0,0,0,0,0]],
                  columns=['fv','c','t','m', 'ytm','erp_type','cpi','inf','er','fe','gp_nc','igp', 'egp_nc'])
df = df.append(new_row, ignore_index=True)

new_row = pd.DataFrame([[1000, 0.05, 3, 1, np.array([0.1, 0.09, 0.105, 0.12]),1,
                         np.array([1.0, 1.045, 1.113, 1.17]),
                         np.array([0.0524, 0.045, 0.0651, 0.0512]),
                         0,0,0,0,0]],
                  columns=['fv','c','t','m', 'ytm','erp_type','cpi','inf','er','fe','gp_nc','igp', 'egp_nc'])
df = df.append(new_row, ignore_index=True)

new_row = pd.DataFrame([[1000, 0.1, 3, 1, np.array([0.1, 0.09, 0.105, 0.12]),2,
                         0,0,
                         0.1,
                         np.array([0.1, 0.093, 0.105, 0.11]),
                         0,0,0]],
                  columns=['fv','c','t','m', 'ytm','erp_type','cpi','inf','er','fe','gp_nc','igp', 'egp_nc'])
df = df.append(new_row, ignore_index=True)

new_row = pd.DataFrame([[1000, 0.06, 3, 1, np.array([0.1, 0.09, 0.105, 0.12]),3,
                         0,0,0,0,
                         100,0.0423,
                         np.array([100,105,110,109])]],
                  columns=['fv','c','t','m', 'ytm','erp_type','cpi','inf','er','fe','gp_nc','igp', 'egp_nc'])
df = df.append(new_row, ignore_index=True)


# In[25]:


df['bond_price'] = df.apply(bond_price_row, axis=1)


# In[26]:


df

