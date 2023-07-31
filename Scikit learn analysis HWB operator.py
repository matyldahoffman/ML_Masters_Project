import pandas as pd
import copy
import numpy as np
#import shap
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
#from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import normalize

# files
filename = "eellh_HWB_01.txt"
filename2 = "eellh_sm.txt"

# Loading Data
df = pd.read_csv(filename, engine='python')
df2 = pd.read_csv(filename2,engine='python')
df.fillna(0, inplace=True)
df2.fillna(0, inplace=True)
df['event_weight'].value_counts()
feature_names = list(df.columns)
feature_names_sm = list(df2.columns)

original_weights = copy.copy(df["event_weight"])
original_weights_sm = copy.copy(df2["event_weight"])

var_list = ['event_weight','sum_event_weights','n_photon','n_electron','n_muon',\
            'rapidity1','E1','Et1','px1','py1','pt1','mass1',\
                'pt2','rapidity2','E2','Et2','px2','py2','mass2',\
                    'eta','pt','phi','rapidity','E','Et','px','py','mass','m_recoil','ang_sep','delta_phi',\
                        'phi1','phi2','eta1','eta2']
    
var_drop = ['event_weight','sum_event_weights','n_photon','n_electron','n_muon',\
            'rapidity1','E1','Et1','px1','py1','pt1','mass1',\
                'pt2','rapidity2','E2','Et2','px2','py2','mass2',\
                    'eta','pt','phi','E','Et','px','py','mass','m_recoil','ang_sep','delta_phi']

df['delta_phi'] = np.where(df['delta_phi'] > np.pi, df['delta_phi']-2*np.pi, df['delta_phi'])
df['delta_phi'] = np.where(df['delta_phi'] < (-1)*np.pi, df['delta_phi']+2*np.pi, df['delta_phi'])
df2['delta_phi'] = np.where(df2['delta_phi'] > np.pi, df2['delta_phi']-2*np.pi, df2['delta_phi'])
df2['delta_phi'] = np.where(df2['delta_phi'] < (-1)*np.pi, df2['delta_phi']+2*np.pi, df2['delta_phi'])

# Create title for plot 
plot_title = 'Training variables = '
i = 0
for var in var_list:
    if var in var_drop:
        continue
    else:
        i += 1
        plot_title += var + ', '
        if i % 8 == 1:
            plot_title  += '\n'
        continue
    
feature_names_list_to_plot = []
for label in feature_names:
    if label not in var_drop:
        feature_names_list_to_plot.append(label)
    else:
        continue


'''
# only normalise the energy and momentum values
var_normalise = ['E','E1','E2','px','py','pt','Et','mass','rapidity','px1','py1','Et1','mass1','rapidity1','px2','pt1','pt2','py2','Et2','mass2','rapidity2']
for kv in var_normalise:
    if kv not in var_drop:
        for i in range(len(df[kv])-1):
            df[kv].iloc[i] = df[kv].iloc[i]/df[kv].max()
'''
#for var in var_list:
x = np.asarray(df.drop(var_drop, axis=1))
x_sm = np.asarray(df2.drop(var_drop, axis=1))

# axis = 1 tells us to drop labels from the columns and not just the index (that would be x=0)
y = np.asarray(df['event_weight'])
y_sm = np.asarray(df2['event_weight'])
# 
# E, px, py
# normalize_value = (value − min_value) / (max_value − min_value)

#x = tf.truediv(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))

# Data pre-processing
trainX, testX = x[:int(0.8*len(x))], x[int(0.8*len(x)):]
trainY, testY = y[:int(0.8*len(y))], y[int(0.8*len(y)):]
trainX_sm, testX_sm = x_sm[:int(0.8*len(x_sm))], x_sm[int(0.8*len(x_sm)):]
trainY_sm, testY_sm = y_sm[:int(0.8*len(y_sm))], y_sm[int(0.8*len(y_sm)):]
# Normalise the event_weights = Y arrays
trainY[np.where(trainY < 0)] = 0
trainY[np.where(trainY > 0)] = 1
testY[np.where(testY < 0)] = 0
testY[np.where(testY > 0)] = 1


#MLPClassifier train the NN with 3 hidden layers with 150, 100 and 50 nodes
mlp_clf = MLPClassifier(random_state=1, solver='adam', hidden_layer_sizes=(64, 128, 2), \
                        max_iter = 120, activation = 'tanh', validation_fraction=0.2, early_stopping=True)

#mlp_clf = MLPClassifier(random_state=1, solver='lbfgs', hidden_layer_sizes=(64, 96, 2), \
#                        max_iter = 120, activation = 'tanh', validation_fraction=0.2, early_stopping=True)
    
history = mlp_clf.fit(trainX, trainY)
#history_sm = mlp_clf.fit(trainX_sm, trainY_sm)

# Compute feature importance scores
results = permutation_importance(mlp_clf, testX, testY, scoring='accuracy')
importance = results.importances_mean
importance_norm = normalize(importance.reshape(1,-1), norm='l1')

#print(feature_names)
sorted_idx = importance_norm.argsort()[::-1]
for idx in sorted_idx:
    for i in idx:
        print(f"{feature_names_list_to_plot[i]}: {importance_norm[0, i]}")
    
# Model evaluation
y_pred = mlp_clf.predict(testX)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

fig = plot_confusion_matrix(mlp_clf, testX, testY, display_labels=mlp_clf.classes_)
fig.figure_.suptitle(r"$c_{H\widetilde{W}B}/\Lambda^{2}=1/TeV^{2}$"+" | Accuracy={:.2f}%".format(accuracy_score(testY, y_pred)*100))
plt.show()

# construct O_NN

def predict_prob(number):
  return [number[0],1-number[0]]
#HB
proba = mlp_clf.predict_proba(x)
#print(proba)
weight_prob = np.array(list(map(predict_prob, proba)))
#print(weight_prob)
O_NN = []
for i in range(len(weight_prob)):
    O_NN.append(weight_prob[i][0]-weight_prob[i][1])
#print(O_NN)

#sm
proba_sm = mlp_clf.predict_proba(x_sm)
weight_prob_sm = np.array(list(map(predict_prob, proba_sm)))
O_NN_sm = []
for i in range(len(weight_prob_sm)):
    O_NN_sm.append(weight_prob_sm[i][0]-weight_prob_sm[i][1])
#print(O_NN_sm)

# plot differential cross section for delta phi and O_NN

# HWB
WEIGHT_VALUE_HB = 0.00165399
SIGNAL_WEIGHTS_HB = df["sum_event_weights"].iloc[-1]
for i in range(len(original_weights)):
    #print(original_weights.iloc[i])
    original_weights.iloc[i] = original_weights.iloc[i]/100000

#SM
TOTAL_CROSS_SECTION_sm = 1.484148e-02
#0.2383
TOTAL_WEIGHT_sm = 1484.148
#23830.2
SIGNAL_WEIGHTS_sm = df2["sum_event_weights"].iloc[-1]
CROSS_SECTION_PER_EVENT_sm = TOTAL_CROSS_SECTION_sm/TOTAL_WEIGHT_sm

var = "delta_phi"
LUMINOSITY = 5e6
fig, axis = plt.subplots(figsize =(20, 10))
bins = 20

#   PLOTTING DELTA PHI

# SMEFT HB = 0.1
n_err = np.sqrt(np.histogram(df[var], bins=bins, weights = (original_weights)**2)[0])
hist_values, bin_edges = np.histogram(df[var], bins, weights = original_weights)
EFT_deltaphi = np.histogram(df[var], bins, weights = LUMINOSITY*original_weights)[0]
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
#axis.bar(bin_centres, 100*hist_values/bin_width, width=bin_width, alpha=0.2, color="blue", label=r"old")
axis.bar(bin_centres, 10*hist_values/bin_width, width=bin_width, alpha=0.2, color="red", label=r"$c_{H\widetilde{W}B}/\Lambda^{2}=1/TeV^{2}$")
axis.errorbar(bin_centres, 10*(hist_values)/bin_width, yerr=10*n_err/bin_width, fmt='o', color="red")

# SM
n_err2 = np.sqrt(np.histogram(df2[var], bins=bins, weights = (CROSS_SECTION_PER_EVENT_sm*original_weights_sm)**2)[0])
hist_values2, bin_edges = np.histogram(df2[var], bins, weights = CROSS_SECTION_PER_EVENT_sm*original_weights_sm)
n_deltaphi = np.histogram(df2[var], bins, weights = LUMINOSITY*CROSS_SECTION_PER_EVENT_sm*original_weights_sm)[0]
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
axis.errorbar(bin_centres, hist_values2/bin_width/10, yerr=n_err2/bin_width/10, fmt='o', color="orange")
axis.bar(bin_centres, hist_values2/bin_width/10, width=bin_width, alpha=0.5, color="orange", label = "SM/10")

plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
axis.set_xlabel('$\Delta \phi$', fontsize = 35)
axis.set_ylabel("$\\frac{d\sigma}{d \Delta \phi}$ [pb/rad]", fontsize = 35)
#plt.title(, fontsize=35)
#axis.set_yscale('log', nonpositive='clip')
axis.legend(fontsize = 25)
axis.plot()
plt.show()

fig, axis = plt.subplots(figsize =(20, 10))


# SMEFT weight arrays
weights = []
weights_squared = []
weight_chi_square_HB = []
for i in range(len(x)):
    weights.append(original_weights.iloc[i])
    weights_squared.append((original_weights.iloc[i])**2)
for i in range(0,len(O_NN)):
    weight_chi_square_HB.append(LUMINOSITY*original_weights.iloc[i])
    
# SM weight arrays
weights_sm = []
weights_sm_squared = []
weight_chi_square_sm = []
for i in range(len(x_sm)):
    weights_sm.append(CROSS_SECTION_PER_EVENT_sm*original_weights_sm.iloc[i])
    weights_sm_squared.append((CROSS_SECTION_PER_EVENT_sm*original_weights_sm.iloc[i])**2)
for i in range(0,len(O_NN_sm)):
    weight_chi_square_sm.append(LUMINOSITY*CROSS_SECTION_PER_EVENT_sm*original_weights_sm.iloc[i])


bins = 20

#   PLOTTING O_NN

# SMEFT
n_err3 = np.sqrt(np.histogram(O_NN, bins=bins, weights = weights_squared)[0])
hist_values3, bin_edges_ONN = np.histogram(O_NN, bins, weights = weights)
EFT_ONN = np.histogram(O_NN, bins, weights = weight_chi_square_HB)[0]
bin_centres_ONN = 0.5*(bin_edges_ONN[1:] + bin_edges_ONN[:-1])
bin_width_ONN = (bin_edges_ONN[-1]-bin_edges_ONN[0])/bins
axis.set_xlim(xmin=-1, xmax = 1)
#axis.bar(bin_centres_ONN, hist_values3*100/bin_width_ONN, width=bin_width_ONN, alpha=0.2, color="red", label=r"$c_{H\widetilde{W}B}/\Lambda^{2}=1/TeV^{2}$")
axis.errorbar(bin_centres_ONN, hist_values3*100/bin_width_ONN, yerr=n_err3*100/bin_width_ONN, fmt='o', color="forestgreen")
first_hist_val = copy.copy(hist_values3[0])
new_hist_values = np.insert(hist_values3,0,first_hist_val,axis=None)
first_bin_val = copy.copy(bin_centres_ONN[0]-bin_width_ONN)
new_bin_centres_ONN = np.insert(bin_centres_ONN,0,first_bin_val,axis=None)
axis.step(new_bin_centres_ONN+bin_width_ONN/2, new_hist_values*100/bin_width_ONN, color="forestgreen", label=r"$c_{H\widetilde{W}B}/\Lambda^{2}=1/TeV^{2}$")
# SM
n_err4 = np.sqrt(np.histogram(O_NN_sm, bins=bins, weights = weights_sm_squared)[0])
hist_values4, bin_edges_ONN = np.histogram(O_NN_sm, bins, weights = weights_sm)
n_ONN = np.histogram(O_NN_sm, bins, weights = weight_chi_square_sm)[0]
bin_centres_ONN = 0.5*(bin_edges_ONN[1:] + bin_edges_ONN[:-1])
bin_width_ONN = (bin_edges_ONN[-1]-bin_edges_ONN[0])/bins
axis.set_xlim(xmin=-1, xmax = 1)
axis.errorbar(bin_centres_ONN, hist_values4/bin_width_ONN, yerr=n_err4/bin_width_ONN, fmt='o', color="orange")
first_hist_val = copy.copy(hist_values4[0])
new_hist_values = np.insert(hist_values4,0,first_hist_val,axis=None)
first_bin_val = copy.copy(bin_centres_ONN[0]-bin_width_ONN)
new_bin_centres_ONN = np.insert(bin_centres_ONN,0,first_bin_val,axis=None)
axis.step(new_bin_centres_ONN+bin_width_ONN/2, new_hist_values/bin_width_ONN, color="orange", label = "SM/10")
#axis.bar(bin_centres_ONN, hist_values4/bin_width_ONN, width=bin_width_ONN, alpha=0.5, color="orange", label = "SM/10")

plt.axhline(0,color='lightgray')

plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
axis.set_xlabel('$O_{NN}$', fontsize = 35)
axis.set_ylabel("$\\frac{d\sigma}{d O_{NN}} [pb/O_{NN}]$", fontsize = 35)
#plt.title(, fontsize=35)
#axis.set_yscale('log', nonpositive='clip')
axis.legend(fontsize = 25)
axis.plot()
plt.show()

def chi_square(sm, interference, c):
    return (10*c*interference)**2/sm

chi_square_deltaphi = []
chi_square_ONN = []

c_deltaphi = np.linspace(-0.07,0.07,len(n_deltaphi))
c_ONN = np.linspace(-0.07,0.07,len(n_ONN))

for c in c_deltaphi:
    chi_sq_dp_val = 0
    for i in range(0,len(n_deltaphi)):
        chi_sq_dp_val += chi_square(n_deltaphi[i],EFT_deltaphi[i],c)
    chi_square_deltaphi.append(chi_sq_dp_val)
for c in c_ONN:
    chi_sq_ONN_val = 0
    for i in range(0,len(n_ONN)):
        chi_sq_ONN_val += chi_square(n_ONN[i],EFT_ONN[i],c)
    chi_square_ONN.append(chi_sq_ONN_val)
       
# CHI SQUARE

fig, axis = plt.subplots(figsize =(20, 10))
axis.plot(c_ONN, chi_square_ONN,label='$O_{NN}$',color='mediumvioletred')
axis.plot(c_deltaphi, chi_square_deltaphi,label='$\Delta\phi$',color='lightseagreen')
axis.set_xlabel(r"$c_{H\widetilde{W}B}/\Lambda^{2}[TeV^{-2}]$", fontsize = 35)
axis.set_ylabel(r"$\Delta \chi^{2}$", fontsize = 35)
axis.legend(fontsize = 25)
axis.plot()
plt.show()

# q_NP

def q_NP(b,s,c):
    return 2*(-(0.75)**2*b*np.log(1+10*c*(0.75)**2*s/((0.75)**2*b))+10*c*(0.75)**2*s)

q_NP_deltaphi = []
for c in c_deltaphi:
    q = 0
    for i in range(len(n_deltaphi)):
        q += q_NP(n_deltaphi[i],EFT_deltaphi[i],c)
    q_NP_deltaphi.append(q)

q_NP_ONN = []
for c in c_ONN:
    q = 0
    for i in range(len(n_ONN)):
        q += q_NP(n_ONN[i],EFT_ONN[i],c)
    q_NP_ONN.append(q)

fig, axis = plt.subplots(figsize =(20, 10))
axis.plot(c_ONN, q_NP_ONN,label='$O_{NN}$',color='mediumvioletred')
axis.plot(c_deltaphi, q_NP_deltaphi,label='$\Delta\phi$',color='lightseagreen')
axis.axhline(1.96**2, label='$q_{c}=1.96^{2}$',linestyle='dashed',color='olive')
plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
axis.set_xlabel(r"$c_{H\widetilde{W}B}/\Lambda^{2}[TeV^{-2}]$", fontsize = 35)
axis.set_ylabel(r"$q^{NP}$", fontsize = 35)
#plt.title(, fontsize=35)
#axis.set_yscale('log', nonpositive='clip')
axis.set_ylim(-0.01,5)

axis.legend(fontsize = 25)
axis.plot()
plt.show()

# Limit extraction

coeffs_deltaphi = np.polyfit(c_deltaphi, q_NP_deltaphi, 2)
coeffs_ONN = np.polyfit(c_ONN, q_NP_ONN, 2)

print(f'delta phi curve = {coeffs_deltaphi[0]}x^2+{coeffs_deltaphi[1]}x+{coeffs_deltaphi[2]}')
print(f'ONN curve = {coeffs_ONN[0]}x^2+{coeffs_ONN[1]}x+{coeffs_ONN[2]}')

deltaphi_lims = (-coeffs_deltaphi[1]+np.sqrt(coeffs_deltaphi[1]**2-4*coeffs_deltaphi[0]*(coeffs_deltaphi[2]-1.96**2)))/(2*coeffs_deltaphi[0])
ONN_lims = (-coeffs_ONN[1]+np.sqrt(coeffs_ONN[1]**2-4*coeffs_ONN[0]*(coeffs_ONN[2]-1.96**2)))/(2*coeffs_ONN[0])

print('Limiting value of c for delta phi = ', deltaphi_lims)
print('Limiting value of c for ONN = ', ONN_lims)