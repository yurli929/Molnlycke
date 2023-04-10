#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import glob

path = '/home/yurli/Molnlycke/case2/csv'
all_files = glob.glob(path + "/*.csv")


# In[3]:


import os

df_names = []
df = {}

for filename in all_files:
    df_name = os.path.splitext(os.path.basename(filename))[0]
    df[df_name] = pd.read_csv(filename, index_col = None, header = 0)
    df_names.append(df_name)
    print(df_name, df[df_name].shape)


# In[4]:


# len(set(df['patients']['Id'])) == df['patients'].shape[0]


# In[5]:


patientID = '76982e06-f8b8-4509-9ca3-65a99c8650fe'


# In[6]:


df['patients']['Name'] = df['patients']['FIRST'] + ' ' + df['patients']['LAST']
zipCode = []
for i in df['patients']['ZIP']:
    try:
        zipCode.append(str(int(i)))
    except:
        zipCode.append('')
df['patients']['ZIP'] = zipCode
df['patients']['Address'] = df['patients']['ADDRESS'] + ', ' + df['patients']['ZIP'] + ' ' + df['patients']['CITY'] + ', ' + df['patients']['STATE'] + ', ' + df['patients']['COUNTY']
df['patients']['MARITAL'] = df['patients']['MARITAL'].fillna('Unknown')


# In[7]:


import dateutil.parser

birth = []
for patient in df['encounters']['PATIENT']:
    birthDate = df['patients'][df['patients']['Id'] == patient]['BIRTHDATE'].values[0]
    birthYear = list(pd.DatetimeIndex([birthDate]).year)
    birth.append(birthYear)
df['encounters']['birthYear'] = np.asarray(birth).reshape(-1)
df['encounters']['encounterYear'] = pd.DatetimeIndex(df['encounters']['START']).year
df['encounters']['age'] = df['encounters']['encounterYear'] - df['encounters']['birthYear']
df['encounters']['START'] = [dateutil.parser.parse(x).strftime("%Y-%m-%d") for x in df['encounters']['START']]
df['encounters']['STOP'] = [dateutil.parser.parse(x).strftime("%Y-%m-%d") for x in df['encounters']['STOP']]


# In[8]:


df['observations']['UNITS'] = df['observations']['UNITS'].fillna('')
df['observations']['DATE'] = [dateutil.parser.parse(x).strftime("%Y-%m-%d %H:%M:%S") for x in df['observations']['DATE']]
df['observations']['observations'] = df['observations']['DATE'] + ': ' + df['observations']['DESCRIPTION'] + ': ' + df['observations']['VALUE'] + ' ' + df['observations']['UNITS']


# In[9]:


df['conditions']['STOP'] = df['conditions']['STOP'].fillna('')


# In[10]:


df['medications']['START'] = [dateutil.parser.parse(x).strftime("%Y-%m-%d") for x in df['medications']['START']]

stop = []
for x in df['medications']['STOP']:
    try:
        stop.append(dateutil.parser.parse(x).strftime("%Y-%m-%d"))
    except:
        stop.append('')
df['medications']['STOP'] = stop


# In[11]:


df['careplans']['STOP'] = df['careplans']['STOP'].fillna('')


# In[12]:


df['immunizations']['DATE'] = [dateutil.parser.parse(x).strftime("%Y-%m-%d") for x in df['immunizations']['DATE']]


# In[13]:


df['procedures']['DATE'] = [dateutil.parser.parse(x).strftime("%Y-%m-%d") for x in df['procedures']['DATE']]


# In[14]:


# df['imaging_studies']['DATE'] = [dateutil.parser.parse(x).strftime("%Y-%m-%d") for x in df['imaging_studies']['DATE']]


# In[15]:


def care_data_collection(patientID):
    encounters_sub = df['encounters'][df['encounters']['PATIENT'] == patientID]   
    observations_sub = df['observations'][df['observations']['PATIENT'] == patientID]   
    conditions_sub = df['conditions'][df['conditions']['PATIENT'] == patientID]    
    medications_sub = df['medications'][df['medications']['PATIENT'] == patientID]    
    careplans_sub = df['careplans'][df['careplans']['PATIENT'] == patientID]      
    immunizations_sub = df['immunizations'][df['immunizations']['PATIENT'] == patientID]   
    procedures_sub = df['procedures'][df['procedures']['PATIENT'] == patientID]
    return encounters_sub, observations_sub, conditions_sub, medications_sub, careplans_sub, immunizations_sub, procedures_sub


# In[16]:


separator = ", "
def patient_Data_printer(patientID):
    name = df['patients'][df['patients']['Id'] == patientID]['Name'].values[0]
    race = df['patients'][df['patients']['Id'] == patientID]['RACE'].values[0]
    ethnicity = df['patients'][df['patients']['Id'] == patientID]['ETHNICITY'].values[0]
    gender = df['patients'][df['patients']['Id'] == patientID]['GENDER'].values[0]
    birthDate = df['patients'][df['patients']['Id'] == patientID]['BIRTHDATE'].values[0]
    ifMarital = df['patients'][df['patients']['Id'] == patientID]['MARITAL'].values[0]
    address = df['patients'][df['patients']['Id'] == patientID]['Address'].values[0]
    
    patient_allergies = 'N/A'
    if df['allergies'][df['allergies']['PATIENT'] == patientID].shape[0] != 0:
        patient_allergies = separator.join(df['allergies'][df['allergies']['PATIENT'] == patientID]['DESCRIPTION'])
    print(name)
    print('============================')
    print('Race: ' + race)
    print('Ethnicity: ' + ethnicity)
    print('Gender: ' + gender)
    print('Birth Date: ' + birthDate)
    print('Marital Status: ' + ifMarital)
    print('Address: ' + address)
    print('==========================================================================================')
    print('Allergies: ' + patient_allergies)
    print('==========================================================================================')  

    
    encounter_sub, observation_sub, conditions_sub, medications_sub, careplans_sub, immunizations_sub, procedures_sub = care_data_collection(patientID)
    
    for encounter_id in encounter_sub['Id']:
        observation_perEncounter = observation_sub[observation_sub['ENCOUNTER'] == encounter_id]
        condition_perEncounter = conditions_sub[conditions_sub['ENCOUNTER'] == encounter_id]
        medication_perEncounter = medications_sub[medications_sub['ENCOUNTER'] == encounter_id]
        careplan_perEncounter = careplans_sub[careplans_sub['ENCOUNTER'] == encounter_id]
        immunization_perEncounter = immunizations_sub[immunizations_sub['ENCOUNTER'] == encounter_id]
        procedure_perEncounter = procedures_sub[procedures_sub['ENCOUNTER'] == encounter_id]
        
        print('Encounter: ')
        print(encounter_sub[encounter_sub['Id'] == encounter_id]['START'].values[0] + ': ' + 
              encounter_sub[encounter_sub['Id'] == encounter_id]['DESCRIPTION'].values[0] + '  (class: ' +
              encounter_sub[encounter_sub['Id'] == encounter_id]['ENCOUNTERCLASS'].values[0] + ')')
        
        if observation_perEncounter.shape[0] != 0:
            print('Observations:')
            print(*observation_perEncounter['observations'], sep = "\n")
        
        if condition_perEncounter.shape[0] != 0:
            print('Condition:')
            print(condition_perEncounter['START'].values[0] + ' -- ' + condition_perEncounter['STOP'].values[0] + ': ' +
                  condition_perEncounter['DESCRIPTION'].values[0])    
            
        if medication_perEncounter.shape[0] != 0:
            print('Medications:')
            print(medication_perEncounter['START'].values[0] + ' -- ' + medication_perEncounter['STOP'].values[0] + ': ' +
                  medication_perEncounter['DESCRIPTION'].values[0])
        
        if careplan_perEncounter.shape[0] != 0:
            print('Care Plans:')
            print(careplan_perEncounter['START'].values[0] + ' -- ' + careplan_perEncounter['STOP'].values[0] + ': ' +
                  careplan_perEncounter['DESCRIPTION'].values[0])
            
        if immunization_perEncounter.shape[0] != 0:
            print('Immunization:')
            print(immunization_perEncounter['DATE'].values[0] + ': ' + immunization_perEncounter['DESCRIPTION'].values[0])
        
        if procedure_perEncounter.shape[0] != 0:
            print('Procedure:')
            print(procedure_perEncounter['DATE'].values[0] + ': ' + procedure_perEncounter['DESCRIPTION'].values[0])
        
        print('------------------------------------------------------------------------------------------')


# In[17]:


patient_Data_printer(patientID)


# In[18]:


encounters, observations, conditions, medications, careplans, immunizations, procedures = care_data_collection(patientID)


# In[19]:


encounters['featureName'] = ['Encounters'] * encounters.shape[0]
conditions['featureName'] = ['Conditions'] * conditions.shape[0]
medications['featureName'] = ['Medications'] * medications.shape[0]
careplans['featureName'] = ['Careplans'] * careplans.shape[0]
immunizations['featureName'] = ['Immunizations'] * immunizations.shape[0]


# In[20]:


immunizations = immunizations.rename(columns = {"DATE": "START"})


# In[21]:


col_toUse = ['START', 'STOP', 'DESCRIPTION', 'featureName']

encounters_toUse = encounters[col_toUse + ['ENCOUNTERCLASS']]
conditions_toUse = conditions[col_toUse]
careplans_toUse = careplans[col_toUse]
medications_toUse = medications[col_toUse]
immunizations_toUse = immunizations[['START', 'DESCRIPTION', 'featureName']]


# In[22]:


df_toUse = pd.concat([medications_toUse, careplans_toUse, conditions_toUse]).fillna('')


# In[23]:


import plotly.express as px
 
fig = px.timeline(df_toUse.sort_values('START'),
                  x_start = "START",
                  x_end = "STOP",
                  y = "featureName",
                  text = "DESCRIPTION",
                  color = "featureName",
                  width = 2000, height = 300)

for i in range(encounters_toUse.shape[0]):
    if encounters_toUse['ENCOUNTERCLASS'].values[i] == 'urgentcare': 
        fig.add_vline(x = encounters_toUse['START'].values[i], line_width = 1, line_dash = "dash", line_color = "orange")
    if encounters_toUse['ENCOUNTERCLASS'].values[i] == 'emergency': 
        fig.add_vline(x = encounters_toUse['START'].values[i], line_width = 1, line_dash = "dash", line_color = "red")
    if encounters_toUse['ENCOUNTERCLASS'].values[i] == 'inpatient': 
        fig.add_vline(x = encounters_toUse['START'].values[i], line_width = 1, line_dash = "dash", line_color = "yellow")
    if encounters_toUse['ENCOUNTERCLASS'].values[i] == 'outpatient': 
        fig.add_vline(x = encounters_toUse['START'].values[i], line_width = 1, line_dash = "dash", line_color = "cyan")
    if encounters_toUse['ENCOUNTERCLASS'].values[i] == 'ambulatory': 
        fig.add_vline(x = encounters_toUse['START'].values[i], line_width = 1, line_dash = "dash", line_color = "lightblue")
    if encounters_toUse['ENCOUNTERCLASS'].values[i] == 'wellness': 
        fig.add_vline(x = encounters_toUse['START'].values[i], line_width = 1, line_dash = "dash", line_color = "lightgreen")    
    
    fig.add_annotation(x = encounters_toUse['START'].values[i], y = 1, yref = "paper", text = encounters_toUse['ENCOUNTERCLASS'].values[i])

fig.add_scatter(x = immunizations_toUse['START'], y = ['Immunizations'] * immunizations_toUse.shape[0], 
                mode = "markers", name = "Immunizations", marker = dict(size = 10))

fig.update_layout(font = dict(family="Courier New, monospace", size = 10))
fig.update_yaxes(title = '', showticklabels = True)
fig.show()


# In[24]:


# remove a patient with condition on multiple encounters
conditions_byPatient = df['conditions'].drop_duplicates(subset = ['PATIENT', 'DESCRIPTION'],
                                                        keep = 'first').reset_index(drop = True)


# In[25]:


condition_rank = pd.DataFrame(conditions_byPatient['DESCRIPTION'].value_counts()).reset_index()
condition_rank.columns = ['Conditions', 'Freq']


# In[26]:


import holoviews as hv
hv.extension('bokeh')

bars = hv.Bars(data = condition_rank)
bars.opts(width = 2000, height = 800, xrotation = 90, title = 'Conditions Rank', ylabel = 'Amount')


# In[27]:


top3 = list(condition_rank.head(3)['Conditions'].values)
print(top3)


# In[28]:


conditions_top3 = conditions_byPatient[conditions_byPatient['DESCRIPTION'].isin(list(condition_rank.head(3)['Conditions'].values))]

race = []
ethnicity = []
gender = []
ifMarital = []

for patient in conditions_top3['PATIENT']:
    race.append(df['patients'][df['patients']['Id'] == patient]['RACE'].values[0])
    ethnicity.append(df['patients'][df['patients']['Id'] == patient]['ETHNICITY'].values[0])
    gender.append(df['patients'][df['patients']['Id'] == patient]['GENDER'].values[0])
    ifMarital.append(df['patients'][df['patients']['Id'] == patient]['MARITAL'].values[0])   

conditions_top3['race'] = race
conditions_top3['ethnicity'] = ethnicity
conditions_top3['gender'] = gender
conditions_top3['ifMarital'] = ifMarital
conditions_top3['ifMarital'] = conditions_top3['ifMarital'].fillna('Unknown')


# In[29]:


conditions_top3_age = df['conditions'][df['conditions']['DESCRIPTION'].isin(top3)]

birthYear = []
for patient in conditions_top3_age['PATIENT']:
    birthDate = df['patients'][df['patients']['Id'] == patient]['BIRTHDATE'].values[0]
    birthYear.append(list(pd.DatetimeIndex([birthDate]).year))

conditions_top3_age['birthYear'] = np.asarray(birthYear).reshape(-1)
conditions_top3_age['conditionYear'] = pd.DatetimeIndex(conditions_top3_age['START']).year
conditions_top3_age['age'] = conditions_top3_age['conditionYear'] - conditions_top3_age['birthYear']


# In[30]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig1 = make_subplots(rows = 1, cols = 3, specs=[[{'type':'domain'}]*3], subplot_titles = (top3))
fig2 = make_subplots(rows = 1, cols = 3, specs=[[{'type':'domain'}]*3], subplot_titles = (top3))
fig3 = make_subplots(rows = 1, cols = 3, specs=[[{'type':'domain'}]*3], subplot_titles = (top3))
fig4 = make_subplots(rows = 1, cols = 3, specs=[[{'type':'domain'}]*3], subplot_titles = (top3))
fig5 = make_subplots(rows = 1, cols = 3, subplot_titles = (top3))

for i in range(3):
    cond_df = conditions_top3[conditions_top3['DESCRIPTION'] == top3[i]]
    gender = pd.DataFrame(cond_df['gender'].value_counts()).reset_index()
    race = pd.DataFrame(cond_df['race'].value_counts()).reset_index()
    ethnicity = pd.DataFrame(cond_df['ethnicity'].value_counts()).reset_index()
    ifMarital = pd.DataFrame(cond_df['ifMarital'].value_counts()).reset_index()
    
    fig1.add_trace(
        go.Pie(values = list(gender['gender'].values), labels = list(gender['index'].values)),
        row = 1, col = (i+1)) 
    fig2.add_trace(
        go.Pie(values = list(race['race'].values), labels = list(race['index'].values)),
        row = 1, col = (i+1)) 
    fig3.add_trace(
        go.Pie(values = list(ethnicity['ethnicity'].values), labels = list(ethnicity['index'].values)),
        row = 1, col = (i+1)) 
    fig4.add_trace(
        go.Pie(values = list(ifMarital['ifMarital'].values), labels = list(ifMarital['index'].values)),
        row = 1, col = (i+1)) 
    fig5.add_trace(
        go.Histogram(x = conditions_top3_age[conditions_top3_age['DESCRIPTION'] == top3[i]]['age'].values,
                     xbins = dict(
                     start = 0, end = 100, size = 10), # M18 stands for 18 months
                     autobinx = False
                     ), 
        row = 1, col = (i+1))

fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.update_layout(width = 1000, height = 450, showlegend = False)
fig5.show()


# In[31]:


encounterClass_eachCase = []
medication_eachCase = []
careplan_eachCase = []
# immunization_eachCase = []

for i in range(conditions_top3_age.shape[0]):
    el = df['encounters'][(df['encounters']['PATIENT'] == conditions_top3_age['PATIENT'].values[i]) 
                          & (df['encounters']['Id'] == conditions_top3_age['ENCOUNTER'].values[i])]  
    encounterClass_eachCase.append(el['ENCOUNTERCLASS'].values[0])
    
    med = df['medications'][(df['medications']['PATIENT'] == conditions_top3_age['PATIENT'].values[i]) 
                            & (df['medications']['ENCOUNTER'] == conditions_top3_age['ENCOUNTER'].values[i])]
    if med.shape[0] != 0:
        medication_eachCase.append(med['DESCRIPTION'].values[0])
    else:
        medication_eachCase.append('None')
        
    cp = df['careplans'][(df['careplans']['PATIENT'] == conditions_top3_age['PATIENT'].values[i]) 
                          & (df['careplans']['ENCOUNTER'] == conditions_top3_age['ENCOUNTER'].values[i])]
    if cp.shape[0] != 0:
        careplan_eachCase.append(cp['DESCRIPTION'].values[0])
    else:
        careplan_eachCase.append('None')
        
    '''
    imm = df['immunizations'][(df['immunizations']['PATIENT'] == conditions_top3_age['PATIENT'].values[i]) 
                              & (df['immunizations']['ENCOUNTER'] == conditions_top3_age['ENCOUNTER'].values[i])]
    if imm.shape[0] != 0:
        immunization_eachCase.append(imm['DESCRIPTION'].values[0])
    else:
        immunization_eachCase.append('None')
    '''


# In[32]:


conditions_top3_age['encounterClass'] = encounterClass_eachCase
conditions_top3_age['medication'] = medication_eachCase
conditions_top3_age['careplan'] = careplan_eachCase
# conditions_top3_age['immunization'] = immunization_eachCase


# In[33]:


# set(immunization_eachCase)


# In[34]:


fig6 = make_subplots(rows = 1, cols = 3, specs=[[{'type':'domain'}]*3], subplot_titles = (top3))
fig7 = make_subplots(rows = 1, cols = 3, specs=[[{'type':'domain'}]*3])
fig8 = make_subplots(rows = 1, cols = 3, specs=[[{'type':'domain'}]*3], subplot_titles = (top3))

for i in range(3):
    cond_df = conditions_top3_age[conditions_top3_age['DESCRIPTION'] == top3[i]]
    ec_count = pd.DataFrame(cond_df['encounterClass'].value_counts()).reset_index()
    med_count = pd.DataFrame(cond_df['medication'].value_counts()).reset_index()
    cp_count = pd.DataFrame(cond_df['careplan'].value_counts()).reset_index()
    
    fig6.add_trace(
        go.Pie(values = list(ec_count['encounterClass'].values), labels = list(ec_count['index'].values)),
        row = 1, col = (i+1)) 
    fig7.add_trace(
        go.Pie(values = list(med_count['medication'].values), labels = list(med_count['index'].values)),
        row = 1, col = (i+1)) 
    fig8.add_trace(
        go.Pie(values = list(cp_count['careplan'].values), labels = list(cp_count['index'].values)),
        row = 1, col = (i+1)) 

fig6.update_layout(title_text = "Encounter Class")
fig6.show()
fig7.update_layout(width = 900, height = 900, title_text = "Medications",
                   legend = dict(title_font_family = "Times New Roman", 
                                 font = dict(size = 10),
                                 orientation = "h"))
fig7.show()
fig8.update_layout(width = 1050, height = 600, title_text = "Care Plans",
                   legend = dict(title_font_family = "Times New Roman", 
                                 font = dict(size = 10)))
fig8.show()


# In[35]:


# df['patients'][df['patients']['DEATHDATE'].notna()]


# In[36]:


df['careplans']['DESCRIPTION'].value_counts()


# In[37]:


from datetime import datetime, timedelta
from collections import OrderedDict

# datetime.strptime(df['careplans']['STOP'].values[0], "%Y-%m-%d")-datetime.strptime(df['careplans']['START'].values[0], "%Y-%m-%d")
careplan_top1 = df['careplans'][df['careplans']['DESCRIPTION'] == 'Respiratory therapy']


# In[38]:


# pd.set_option('display.max_rows', 1000)


# In[39]:


'''
careDays = []
for i in range(careplan_top1.shape[0]):
    try: 
        diff = datetime.strptime(careplan_top1['STOP'].values[i], "%Y-%m-%d") - datetime.strptime(careplan_top1['START'].values[i], "%Y-%m-%d")
        careDays.append(diff)
    except:
        careDays.append(np.nan)
careplan_top1['careDays'] = careDays
np.mean(careplan_top1['careDays'])
'''


# In[40]:


careplan_top1 = careplan_top1[careplan_top1['STOP'] != '']


# In[41]:


monthList = []
for i in range(careplan_top1.shape[0]):
    try: 
        diff = datetime.strptime(careplan_top1['STOP'].values[i], "%Y-%m-%d") - datetime.strptime(careplan_top1['START'].values[i], "%Y-%m-%d")
        monthList.append(diff)
    except:
        monthList.append(np.nan)
    
    start = datetime.strptime(careplan_top1['START'].values[i], "%Y-%m-%d")
    end = datetime.strptime(careplan_top1['STOP'].values[i], "%Y-%m-%d")
    mon = OrderedDict(((start + timedelta(_)).strftime(r"%b%Y"), None) for _ in range((end - start).days)).keys()
    monthList = monthList + list(mon)


# In[42]:


monthList = [x for x in monthList if len(str(x)) == 7]


# In[43]:


amount_byMonth = pd.DataFrame(monthList, columns = ['CareDate']).value_counts().reset_index()
amount_byMonth.columns = ['CareDate', 'Amount']
amount_byMonth['Year'] = [datetime.strptime(x, '%b%Y').year for x in amount_byMonth['CareDate']]
amount_byMonth['Month'] = [datetime.strptime(x, '%b%Y').month for x in amount_byMonth['CareDate']]


# In[47]:


start = datetime.strptime('2005-01-01', "%Y-%m-%d")
end = datetime.strptime('2020-04-30', "%Y-%m-%d")
mon = list(OrderedDict(((start + timedelta(_)).strftime(r"%b%Y"), None) for _ in range((end - start).days)).keys())


# In[48]:


amount = pd.DataFrame(mon, columns = ['CareDate']).merge(amount_byMonth[['CareDate', 'Amount']], how = 'left', on = 'CareDate')
amount = amount.fillna(0)


# In[52]:


mu = amount['Amount'].mean()
sd = amount['Amount'].std()

amount_norm = amount.copy()

# Normalize data
amount_norm['Amount'] = (amount['Amount'] - mu) / sd


# In[53]:


train_days = 6 #months
x = []
y = []
num = 0
for i in range(train_days, amount_norm.shape[0]):
    x.append(list(amount_norm['Amount'][num:i].values))
    y.append(amount_norm['Amount'][i])
    num += 1


# In[54]:


X = np.expand_dims(np.array(x), -1)
Y = np.array(y)


# In[72]:


train_num = X.shape[0]-12
X_train = X[0:train_num]
Y_train = Y[0:train_num]

X_test = X[train_num:X.shape[0]]
Y_test = Y[train_num:Y.shape[0]]


# In[56]:


from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.regularizers import l1, l2


# In[73]:


seq_input = Input(shape = (X_train.shape[1], X_train.shape[2]))

x = LSTM(128, kernel_regularizer = l2(0.002), recurrent_regularizer = l2(0.002), bias_regularizer = l2(0.002),
         return_sequences = False)(seq_input)
x = Dropout(0.2)(x)
out = Dense(1, activation = 'linear')(x)
net = Model(seq_input, out)
net.summary()


# In[74]:


net.compile(loss = 'mse', optimizer = Adam(0.001))
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10) 
bm = ModelCheckpoint('../net_weights.hdf5', save_best_only = True, 
                     monitor = 'val_loss', mode = 'min')
net.fit(X_train, Y_train, epochs = 200, batch_size = 8, validation_split = 0.1, callbacks = [es, bm])


# In[75]:


net.load_weights(filepath = '../net_weights.hdf5')
pred = net.predict(X_test)


# In[76]:


result = pd.DataFrame({'CareDate':list(amount_norm['CareDate'][train_num:X.shape[0]].values), 'RealValue': Y_test * sd + mu,'Prediction': (pred * sd + mu).reshape(-1)})


# In[78]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(50, 12))

ax.plot(result['CareDate'], result['Prediction'], linewidth = 3, marker = 'o', markersize = 10, label = 'prediction')
ax.plot(result['CareDate'], result['RealValue'], linewidth = 3, marker = 'o', markersize = 10, label = 'real value')
ax.set_xlabel('Date', fontsize = 30)
ax.set_ylabel('Amount of Cares', fontsize = 30)
ax.tick_params(axis = 'x', labelsize = 25)
ax.tick_params(axis = 'y', labelsize = 25)
ax.legend(fontsize = 30)
ax.grid(True)


# In[82]:


patientD = df['patients'][df['patients']['DEATHDATE'].notna()]
patientA = df['patients'][df['patients']['DEATHDATE'].isna()]


# In[89]:


patientA_sub = patientA.sample(n = patientD.shape[0], replace = False, random_state = 2)
patientA_sub['DEATHDATE'] = patientA_sub['DEATHDATE'].fillna('2023-04-10')


# In[192]:


patientA_sub['ifDead'] = [0] * patientD.shape[0]
patientD['ifDead'] = [1] * patientD.shape[0]
patient_toUse = pd.concat([patientA_sub, patientD])


# In[193]:


patient_col = ['Id', 'BIRTHDATE', 'DEATHDATE', 'LAT', 'LON', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'ifDead']
patient_toUse = patient_toUse[patient_col].reset_index(drop = True)
patient_toUse['age'] = pd.DatetimeIndex(patient_toUse['DEATHDATE']).year - pd.DatetimeIndex(patient_toUse['BIRTHDATE']).year


# In[145]:


# encounterClass = list(set(df['encounters']['ENCOUNTERCLASS']))
# encounterClass


# In[178]:


encounter_count = []
medication_dispenses = []
immunization_record = []
care_record = []

for patient in patient_toUse['Id']:
    encounter = []
    for ec in list(set(df['encounters']['ENCOUNTERCLASS'])):
        ec_count = df['encounters'][df['encounters']['PATIENT'] == patient][df['encounters']['ENCOUNTERCLASS'] == ec]
        encounter.append(ec_count.shape[0])
    encounter_count.append(encounter)
    
    med_info = df['medications'][df['medications']['PATIENT'] == patient]
    if med_info.shape != 0:
        medication_dispenses.append(sum(med_info['DISPENSES'].values))
    else:
        medication_dispenses.append(0)
    
    imm_info = df['immunizations'][df['immunizations']['PATIENT'] == patient]
    immunization_record.append(imm_info.shape[0])
    
    care_info = df['careplans'][df['careplans']['PATIENT'] == patient]
    if care_info.shape[0] == 0:
        care_record.append((0, 0))
    else:
        care_info_sub = care_info[care_info['STOP'] != '']
        care_days = sum([(datetime.strptime(care_info_sub['STOP'].values[k], "%Y-%m-%d")-datetime.strptime(care_info_sub['START'].values[k], "%Y-%m-%d")).days for k in range(care_info_sub.shape[0])]) 
        care_record.append((care_info[care_info['STOP'] == ''].shape[0], care_days))


# In[194]:


patient_toUse['medicationDispenses'] = medication_dispenses
patient_toUse['immunizationRecord'] = immunization_record
patient_toUse = patient_toUse.join(pd.DataFrame(encounter_count, columns = list(set(df['encounters']['ENCOUNTERCLASS']))))
patient_toUse = patient_toUse.join(pd.DataFrame(care_record, columns = ['longtermCareplan(times)', 'shorttermCareplan(days)']))


# In[195]:


col_toDrop = ['Id', 'BIRTHDATE', 'DEATHDATE']
patient_toUse = patient_toUse.drop(col_toDrop, axis = 1)


# In[196]:


to_predict = ['ifDead']
x_cols = [x for x in list(patient_toUse) if x not in to_predict]


# In[198]:


mu = patient_toUse[x_cols].mean(0)
sd = patient_toUse[x_cols].std(0)

# Normalize data
X = (patient_toUse[x_cols] - mu) / sd
Y = patient_toUse[to_predict]


# In[200]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)


# In[226]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import random

# random forest
rv = np.random.randint(0, 10, x.shape[0])
clf = RandomForestClassifier(n_estimators = 100, max_depth = 50)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# neural networks
layer_size = [128, 512, 64]

inp = Input(x_train.shape[1:])
out = inp
for ls in layer_size:
    out = Dense(ls, activation = "relu")(out)
    out = Dropout(0.2)(out)
out = Dense(1, activation = "sigmoid")(out)

    
net = Model(inp, out)
net.compile(loss = "binary_crossentropy", optimizer = Adam(0.001), metrics = ['accuracy'])

mcp_save = ModelCheckpoint('../weights.hdf5', save_best_only = True, monitor = 'val_accuracy', mode = 'max')
     #callbacks=[mcp_save],  validation_split=0.15,
net.fit(x_train, y_train, epochs = 250, batch_size = 16, verbose = 0, validation_split = 0.1, callbacks = [mcp_save])
test_loss, test_acc = net.evaluate(x_test, y_test)

# random guess (baseline)
y_guess = random.choices([0, 1], [0.5, 0.5], k = len(y_test))


# In[227]:


print("RF Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("NN Accuracy:", test_acc)
print("RG Accuracy:", metrics.accuracy_score(y_test, y_guess))


# In[228]:


import shap
shap.initjs()

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(x_train, approximate = False, check_additivity = False)

shap.summary_plot(shap_values[1], x_train)

