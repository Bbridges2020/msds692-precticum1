#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries to run Apriori and visualizations
import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np 
import matplotlib.pyplot as plt
import xlsxwriter
import matplotlib.cm as pltc
import networkx as nx
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


# In[2]:


#Import data set 
floyds = pd.read_csv('/Users/brynnbridges/Desktop/Floyds_data/FloydsData2019-20.txt', error_bad_lines=False)


# In[3]:


floyds.head()


# In[4]:


floyds.info()


# In[5]:


floyds.shape


# In[6]:


floyds.columns


# In[7]:


# cleaning transactions to be all lowercase 
floyds["transactions"].str.lower()


# In[8]:


#using Fuzzy wuzzy to merge fields that are rang out in different ways. 
fuzz.token_set_ratio('floyds_cut, floyds_cut_&_shampoo', 'floyds_cut')


# In[9]:


#View Top services sold & Top Products sold & Top clients per state


# In[10]:


# Arizona
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'az') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by AZ')


# In[11]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'az') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in AZ')


# In[12]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'az']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID AZ')


# In[13]:


# California
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'ca') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by CA')


# In[14]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'ca') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in CA')


# In[15]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'ca']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID CA')


# In[16]:


# Colorado
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'co') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by CO')


# In[17]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'co') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in CO')


# In[18]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'co']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID CO')


# In[19]:


# Florida
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'fl') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by FL')


# In[20]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'fl') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in FL')


# In[21]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'fl']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID FL')


# In[22]:


# Illinois
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'il') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by IL')


# In[23]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'il') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in IL')


# In[24]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'il']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID IL')


# In[25]:


# Kentucky
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'ky') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by KY')


# In[26]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'ky') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in KY')


# In[27]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'ky']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID KY')


# In[28]:


# Maryland
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'md') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by MD')


# In[29]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'md') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in MD')


# In[30]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'md']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID MD')


# In[31]:


# Massachusetts 
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'ma') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by MA')


# In[32]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'ma') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in MA')


# In[33]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'ma']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID MA')


# In[34]:


# Michigan
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'mi') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by MI')


# In[35]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'mi') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in MI')


# In[36]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'mi']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID MI')


# In[37]:


# Minnesota 
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'mn') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by MN')


# In[38]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'mn') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in MN')


# In[39]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'mn']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID MN')


# In[40]:


# Pensylvania 
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'pa') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by PA')


# In[41]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'pa') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in PA')


# In[42]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'pa']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID PA')


# In[43]:


# Texas
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'tx') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by TX')


# In[44]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'tx') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in TX')


# In[45]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'tx']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID TX')


# In[46]:


# Virginia 
fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'va') & (floyds['type'].str.lower() == 'service')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Service Sales by VA')


# In[47]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[(floyds['state'].str.lower() == 'va') & (floyds['type'].str.lower() == 'product')]['transactions'].value_counts().iloc[:20].plot(kind='bar', title='Products sold in VA')


# In[48]:


fig = plt.figure(figsize=(12, 4))
f1 = fig.add_subplot(121)
g = floyds[floyds['state'].str.lower() == 'va']['clientid'].value_counts().iloc[1:20].plot(kind='bar', title='Top Sales by ClientID VA')


# In[49]:


#Focusing on "Floyds cuts" we eliminate other services that are duplicated. 


# In[50]:


#removing transaction for beard trims "secondary service that can be eliminated"
floyds = floyds[~floyds['transactions'].str.contains('basic_beard_trim')]


# In[51]:


floyds = floyds[~floyds['transactions'].str.contains('beard_trim')]


# In[52]:


floyds = floyds[~floyds['transactions'].str.contains('buzz_cut')]


# In[53]:


#removing transaction that are service redos ("r)"
floyds = floyds[~floyds['transactions'].str.contains('r_floyds_cut')]


# In[88]:


#removing message and shampoo since is almost a 1:1 corelation(99% always purchased with Floyds cut)
floyds = floyds[~floyds['transactions'].str.contains('massage_shampoo')]


# In[89]:


floyds = floyds[~floyds['transactions'].str.contains('floyds_cut_&_shampoo')]


# In[90]:


#First we need to create a basket for each state


# In[91]:


# Arizona
basket_az = (floyds[floyds["state"] =="az"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[92]:


basket_az


# In[93]:


# California 
basket_ca = (floyds[floyds["state"] =="ca"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[94]:


# Colorado 
basket_co = (floyds[floyds["state"] =="co"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[95]:


# Florida
basket_fl = (floyds[floyds["state"] =="fl"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[96]:


# Illinois 
basket_il = (floyds[floyds["state"] =="il"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[97]:


# Kentucky 
basket_ky = (floyds[floyds["state"] =="ky"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[98]:


# Maryland
basket_md = (floyds[floyds["state"] =="md"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[99]:


# Massachusetts
basket_ma = (floyds[floyds["state"] =="ma"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[100]:


# Michigan 
basket_mi = (floyds[floyds["state"] =="mi"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[101]:


# Minnesota
basket_mn = (floyds[floyds["state"] =="mn"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[102]:


# Pennsylvania
basket_pa = (floyds[floyds["state"] =="pa"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[103]:


# Texas 
basket_tx = (floyds[floyds["state"] =="tx"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[104]:


# Virginia 
basket_va = (floyds[floyds["state"] =="va"] 
          .groupby(['clientid', 'transactions'])['transactions'] 
          .count().unstack().reset_index().fillna(0) 
          .set_index('clientid'))


# In[105]:


#create frequent items sets using Apriori 
#Produce the association rules
#Visualize the association matrix 
#visualize scatterplot 


# In[106]:


#defining the hot encoding function to make the data suitable for the concerned libraries
def hot_encode(x): 
    if(x<= 0): 
        return 0
    if(x>= 1): 
        return 1


# In[107]:


# Arizona
basket_encoded = basket_az.applymap(hot_encode) 
basket_az = basket_encoded


# In[108]:


frq_items = apriori(basket_az, min_support = 0.002, use_colnames = True)


# In[109]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[110]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()


# In[111]:


draw_graph(rules, 5)


# In[112]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[113]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[114]:


# California 
basket_encoded = basket_ca.applymap(hot_encode) 
basket_ca = basket_encoded


# In[115]:


frq_items_ca = apriori(basket_ca, min_support = 0.01, use_colnames = True)


# In[116]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[117]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()


# In[118]:


draw_graph(rules, 5)


# In[119]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[120]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[121]:


# Colorado 
basket_encoded = basket_co.applymap(hot_encode) 
basket_co = basket_encoded


# In[122]:


frq_items = apriori(basket_co, min_support = 0.002, use_colnames = True)


# In[123]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[124]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()


# In[125]:


draw_graph(rules, 5)


# In[126]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[127]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[128]:


# Florida 
basket_encoded = basket_fl.applymap(hot_encode) 
basket_fl = basket_encoded


# In[129]:


frq_items = apriori(basket_fl, min_support = 0.002, use_colnames = True)


# In[130]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[131]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()


# In[132]:


draw_graph(rules, 5)


# In[133]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[134]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[135]:


# Illinois
basket_encoded = basket_il.applymap(hot_encode) 
basket_il = basket_encoded


# In[ ]:


frq_items = apriori(basket_il, min_support = 0.0001, use_colnames = True)


# In[ ]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[138]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()

draw_graph(rules, 5)


# In[139]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[140]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[141]:


# Kentucky 
basket_encoded = basket_ky.applymap(hot_encode) 
basket_ky = basket_encoded


# In[142]:


frq_items = apriori(basket_ky, min_support = 0.002, use_colnames = True)


# In[143]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[144]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()

draw_graph(rules, 5)


# In[145]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[146]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[147]:


# Maryland 
basket_encoded = basket_md.applymap(hot_encode) 
basket_md = basket_encoded


# In[148]:


frq_items = apriori(basket_md, min_support = 0.002, use_colnames = True)


# In[149]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[150]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()

draw_graph(rules, 5)


# In[151]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[152]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[153]:


# Massachusetts 
basket_encoded = basket_ma.applymap(hot_encode) 
basket_ma = basket_encoded


# In[154]:


frq_items = apriori(basket_ma, min_support = 0.002, use_colnames = True)


# In[155]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[156]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()

draw_graph(rules, 5)


# In[157]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[158]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[159]:


# Michigan
basket_encoded = basket_mi.applymap(hot_encode) 
basket_mi = basket_encoded


# In[160]:


frq_items = apriori(basket_mi, min_support = 0.002, use_colnames = True)


# In[161]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[162]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()

draw_graph(rules, 5)


# In[163]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[164]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[165]:


# Minnesota
basket_encoded = basket_mn.applymap(hot_encode) 
basket_mn = basket_encoded


# In[166]:


frq_items = apriori(basket_mn, min_support = 0.002, use_colnames = True)


# In[167]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[168]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()

draw_graph(rules, 5)


# In[169]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[170]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[171]:


# Pennsylvania
basket_encoded = basket_pa.applymap(hot_encode) 
basket_pa = basket_encoded


# In[172]:


frq_items = apriori(basket_pa, min_support = 0.002, use_colnames = True)


# In[173]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[174]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()

draw_graph(rules, 5)


# In[175]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[176]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[177]:


# Texas 
basket_encoded = basket_tx.applymap(hot_encode) 
basket_tx = basket_encoded


# In[178]:


frq_items = apriori(basket_tx, min_support = 0.002, use_colnames = True)


# In[179]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[180]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()

draw_graph(rules, 5)


# In[181]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[182]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[183]:


# Virginia
basket_encoded = basket_va.applymap(hot_encode) 
basket_va = basket_encoded


# In[184]:


frq_items = apriori(basket_va, min_support = 0.002, use_colnames = True)


# In[185]:


rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head()


# In[186]:


def draw_graph(rules, rules_to_show):  
    G1 = nx.DiGraph()
   
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
#     strs=[]   
   
   
    for i in range (rules_to_show): 
        print(i, rules.iloc[i]['antecedents'])
        G1.add_nodes_from(["R"+str(i)])
    
     
        for a in rules.iloc[i]['antecedents']:
            print(i, rules.iloc[i]['antecedents'])

            G1.add_nodes_from([a])

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            print(c)

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)       

 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.007
    nx.draw_networkx_labels(G1, pos)
    plt.show()

draw_graph(rules, 5)


# In[187]:


support=rules['support'].values
confidence=rules['confidence'].values


# In[188]:


import random
for i in range (len(support)):
    support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




