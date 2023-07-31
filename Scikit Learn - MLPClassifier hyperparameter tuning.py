import pandas as pd
import copy
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

# files
filename = "eellh_HB_01_recoil.txt"
filename2 = "eellh_sm_recoil.txt"

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
print(plot_title)
    
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
#trainY_sm[np.where(trainY_sm < 0)] = 0
#trainY_sm[np.where(trainY_sm > 0)] = 1
#testY_sm[np.where(testY_sm < 0)] = 0
#testY_sm[np.where(testY_sm > 0)] = 1

print(feature_names_list_to_plot)
print(" x = ", x)
print("x train = ", trainX)
print("x test = ", testX)

#scaler = StandardScaler().fit(trainX)
#trainX = scaler.transform(trainX)
#testX = scaler.transform(testX)
layer1_nodes = [32,64,96,128]
layer2_nodes = [32,64,96,128]
activation_functions = ['identity','logistic','tanh','relu']
optimizers = ['adam','sgd','lbfgs']

print("{0:8}{1:8}{2:11}{3:10}{4:9}".format('LAYER 1 ','LAYER 2 ','Activation ','Optimizer ','Accuracy '))
#print(f"{layer1_nodes:7}{:7}{2:10}{3:9}{4:8}".format(layer1_nodes,layer2_nodes,'Activation','Optimizer','Accuracy'))

for n1 in layer1_nodes:
    for n2 in layer2_nodes:
        for af in activation_functions:
                for opt in optimizers:
                    #MLPClassifier train the NN with 3 hidden layers with 150, 100 and 50 nodes
                    mlp_clf = MLPClassifier(random_state=1, solver=opt, hidden_layer_sizes=(n1, n2, 2), \
                                    max_iter = 100, activation = af, validation_fraction=0.2, early_stopping=True)
                        
                    history = mlp_clf.fit(trainX, trainY)
                    y_pred = mlp_clf.predict(testX)
                    print('{0:8}{1:8}{2:11}{3:10}{4:9}'.format(n1,n2,af,opt,accuracy_score(testY, y_pred)))
        
#history_sm = mlp_clf.fit(trainX_sm, trainY_sm)

# Compute feature importance scores
results = permutation_importance(mlp_clf, testX, testY, scoring='accuracy')
importance = results.importances_mean

print(feature_names)
sorted_idx = importance.argsort()[::-1]
for idx in sorted_idx:
    print(f"{feature_names_list_to_plot[idx]}: {importance[idx]}")
    
# Model evaluation
y_pred = mlp_clf.predict(testX)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

fig = plot_confusion_matrix(mlp_clf, testX, testY, display_labels=mlp_clf.classes_)
fig.figure_.suptitle("Confusion Matrix")
plt.show()

# construct O_NN

def predict_prob(number):
  return [number[0],1-number[0]]
#HB
proba = mlp_clf.predict_proba(testX)
#print(proba)
weight_prob = np.array(list(map(predict_prob, proba)))
#print(weight_prob)
O_NN = []
for i in range(len(weight_prob)):
    O_NN.append(weight_prob[i][0]-weight_prob[i][1])
#print(O_NN)

#sm
proba_sm = mlp_clf.predict_proba(testX_sm)
weight_prob_sm = np.array(list(map(predict_prob, proba_sm)))
O_NN_sm = []
for i in range(len(weight_prob_sm)):
    O_NN_sm.append(weight_prob_sm[i][0]-weight_prob_sm[i][1])
#print(O_NN_sm)

# plot differential cross section for delta phi and O_NN

# HB
TOTAL_CROSS_SECTION_HB = 8.872e-08
TOTAL_WEIGHT_HB = 3.47475 
WEIGHT_VALUE_HB = 0.0129655
CROSS_SECTION_PER_EVENT_HB = TOTAL_CROSS_SECTION_HB/TOTAL_WEIGHT_HB
SIGNAL_WEIGHTS_HB = df["sum_event_weights"].iloc[-1]
for i in range(len(original_weights)):
    #print(original_weights.iloc[i])
    original_weights.iloc[i] = original_weights.iloc[i]/100000

#SM
TOTAL_CROSS_SECTION_sm = 1.484148e-02
#0.2383
TOTAL_WEIGHT_sm = 1484.15
#23830.2
SIGNAL_WEIGHTS_sm = df2["sum_event_weights"].iloc[-1]
CROSS_SECTION_PER_EVENT_sm = TOTAL_CROSS_SECTION_sm/TOTAL_WEIGHT_sm

var = "delta_phi"
LUMINOSITY = 5e6
fig, axis = plt.subplots(figsize =(20, 10))
bins = 20

# HB = 0.1
n_err = np.sqrt(np.histogram(df[var], bins=bins, weights = (original_weights)**2)[0])
hist_values, bin_edges = np.histogram(df[var], bins, weights = original_weights)
lambda_deltaphi = np.histogram(df[var], bins, weights = LUMINOSITY*CROSS_SECTION_PER_EVENT_HB*df["event_weight"])[0]
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
axis.bar(bin_centres, 100*hist_values/bin_width, width=bin_width, alpha=0.2, color="red", label="HB = 0.1 * 100")
#axis.errorbar(bin_centres, (hist_values)*(-1), yerr=np.sqrt(100000), fmt='o', color="red", label = "HB")
# SM
n_err2 = np.sqrt(np.histogram(df2[var], bins=bins, weights = (CROSS_SECTION_PER_EVENT_sm*original_weights_sm)**2)[0])
hist_values2, bin_edges = np.histogram(df2[var], bins, weights = CROSS_SECTION_PER_EVENT_sm*original_weights_sm)
n_deltaphi = np.histogram(df2[var], bins, weights = LUMINOSITY*CROSS_SECTION_PER_EVENT_sm*df2["event_weight"])[0]
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
axis.errorbar(bin_centres, hist_values2/bin_width, yerr=n_err2, fmt='o', color="orange")
axis.bar(bin_centres, hist_values2/bin_width, width=bin_width, alpha=0.5, color="orange", label = "SM")

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

print("length of O_NN = ", len(O_NN))

weights = []
weights_squared = []
weight_chi_square_HB = []
for i in range(len(trainX),len(x)):
    weights.append(original_weights.iloc[i])
    weights_squared.append((original_weights.iloc[i])**2)
for i in range(0,len(O_NN)):
    weight_chi_square_HB.append(LUMINOSITY*CROSS_SECTION_PER_EVENT_HB*original_weights.iloc[i])
    
weights_sm = []
weights_sm_squared = []
weight_chi_square_sm = []
for i in range(len(trainX_sm),len(x_sm)):
    weights_sm.append(CROSS_SECTION_PER_EVENT_sm*original_weights_sm.iloc[i])
    weights_sm_squared.append((CROSS_SECTION_PER_EVENT_sm*original_weights_sm.iloc[i])**2)
for i in range(0,len(O_NN_sm)):
    weight_chi_square_sm.append(LUMINOSITY*CROSS_SECTION_PER_EVENT_sm*original_weights_sm.iloc[i])

bins = 10
# O_NN
n_err3 = np.sqrt(np.histogram(O_NN, bins=bins, weights = weights_squared)[0])
hist_values3, bin_edges_ONN = np.histogram(O_NN, bins, weights = weights)
lambda_ONN = np.histogram(O_NN, bins, weights = weight_chi_square_HB)[0]
bin_centres_ONN = 0.5*(bin_edges_ONN[1:] + bin_edges_ONN[:-1])
bin_width_ONN = (bin_edges_ONN[-1]-bin_edges_ONN[0])/bins
axis.set_xlim(xmin=-1, xmax = 1)
axis.bar(bin_centres_ONN, hist_values3*1000/bin_width_ONN, width=bin_width_ONN, alpha=0.2, color="red", label="HB = 0.1*1000")
#axis.errorbar(bin_centres, (hist_values)*(-1), yerr=np.sqrt(100000), fmt='o', color="red", label = "HB")
# SM
n_err4 = np.sqrt(np.histogram(O_NN_sm, bins=bins, weights = weights_sm_squared)[0])
hist_values4, bin_edges_ONN = np.histogram(O_NN_sm, bins, weights = weights_sm)
n_ONN = np.histogram(O_NN_sm, bins, weights = weight_chi_square_sm)[0]
bin_centres_ONN = 0.5*(bin_edges_ONN[1:] + bin_edges_ONN[:-1])
bin_width_ONN = (bin_edges_ONN[-1]-bin_edges_ONN[0])/bins
axis.set_xlim(xmin=-1, xmax = 1)
axis.errorbar(bin_centres_ONN, hist_values4/bin_width_ONN, yerr=n_err3, fmt='o', color="orange")
axis.bar(bin_centres_ONN, hist_values4/bin_width_ONN, width=bin_width_ONN, alpha=0.5, color="orange", label = "SM")

plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
axis.set_xlabel('$O_{NN}$', fontsize = 35)
axis.set_ylabel("$\\frac{d\sigma}{d O_{NN}} [pb/O_{NN}]$", fontsize = 35)
#plt.title(, fontsize=35)
#axis.set_yscale('log', nonpositive='clip')
axis.legend(fontsize = 25)
axis.plot()
plt.show()

def chi_square(sm, interference):
    return (sm-interference)**2/sm

chi_square_deltaphi = []
chi_square_ONN = []
print("length of n_deltaphi = ", len(n_deltaphi))
print("length of lambda_deltaphi = ", len(lambda_deltaphi))
print("length of n_ONN = ", len(n_ONN))
print("length of lambda_ONN = ", len(lambda_ONN))

for i in range(0,len(n_deltaphi)):
    chi_square_deltaphi.append(chi_square(n_deltaphi[i],lambda_deltaphi[i]))
    if i < len(n_ONN):
        chi_square_ONN.append(chi_square(n_ONN[i],lambda_ONN[i]))

# chi square plots
fig, axis = plt.subplots(figsize =(20, 10))
axis.plot(bin_centres_ONN, chi_square_ONN,label='$O_{NN}$',color='orange')
axis.plot(bin_centres, chi_square_deltaphi,label='$\Delta\phi$',color='r')
plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
#axis.set_xlabel(, fontsize = 35)
axis.set_ylabel(r"$\chi^{2}$", fontsize = 35)
#plt.title(, fontsize=35)
#axis.set_yscale('log', nonpositive='clip')
axis.legend(fontsize = 25)
axis.plot()
plt.show()

# Limits setting curve

def q_NP(b,s,c):
    return 2*(-b*np.log(1+10*c*s/b)+c*s)

q_NP_deltaphi = []
c_deltaphi = np.linspace(0,52,len(n_deltaphi))
for c in c_deltaphi:
    q = 0
    for i in range(len(n_deltaphi)):
        q += q_NP(n_deltaphi[i],lambda_deltaphi[i],c)
    q_NP_deltaphi.append(q)

q_NP_ONN = []
c_ONN = np.linspace(0,52,len(n_ONN))
for c in c_ONN:
    q = 0
    for i in range(len(n_ONN)):
        q += q_NP(n_ONN[i],lambda_ONN[i],c)
    q_NP_ONN.append(q*10000000000)

fig, axis = plt.subplots(figsize =(20, 10))
axis.plot(c_ONN, q_NP_ONN,label='$O_{NN}*10000000000$',color='orange')
axis.plot(c_deltaphi, q_NP_deltaphi,label='$\Delta\phi$',color='r')
axis.axhline(1.96**2, label='$q_{c}=1.96^{2}$')
plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
axis.set_xlabel('c', fontsize = 35)
axis.set_ylabel(r"$q^{NP}$", fontsize = 35)
#plt.title(, fontsize=35)
#axis.set_yscale('log', nonpositive='clip')
axis.legend(fontsize = 25)
axis.plot()
plt.show()

