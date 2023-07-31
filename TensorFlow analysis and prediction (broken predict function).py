# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:46:58 2023

Reconstructing delta phi

@author: matyl
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import copy

# files
filename = "eellh_HB_01.txt"
filename2 = "eellh_sm.txt"

# Loading Data
df = pd.read_csv(filename, engine='python')
df2 = pd.read_csv(filename2,engine='python')
df.fillna(0, inplace=True)
df2.fillna(0, inplace=True)
df['event_weight'].value_counts()
feature_names = list(df.columns)

original_weights = copy.copy(df["event_weight"])
original_weights_sm = copy.copy(df2["event_weight"])
    
var_list = ['event_weight','sum_event_weights','n_photon','n_electron','n_muon',\
            'rapidity1','E1','Et1','px1','py1','pt1','mass1',\
                'pt2','rapidity2','E2','Et2','px2','py2','mass2',\
                    'eta','pt','phi','rapidity','E','Et','px','py','mass','m_recoil','ang_sep','delta_phi',\
                        'phi1','phi2','eta1','eta2']
    
var_drop = ['event_weight','sum_event_weights','n_photon','n_electron','n_muon',\
            'rapidity1','E1','Et1','px1','py1','pt1','mass1','eta1',\
                'pt2','rapidity2','E2','Et2','px2','py2','mass2',\
                    'rapidity','eta','pt','phi','E','Et','px','py','mass','m_recoil','ang_sep','delta_phi']

    
#var_not_drop = ['eta','pt','phi','rapidity','E','Et','px','py','mass','m_recoil','ang_sep','delta_phi']

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
#print(plot_title)
    
feature_names_list_to_plot = []
for label in feature_names:
    if label not in var_drop:
        if label == 'phi1':
            feature_names_list_to_plot.append('$\phi_{1}$')
        elif label == 'phi2':
            feature_names_list_to_plot.append('$\phi_{2}$')
        elif label == 'eta1':
            feature_names_list_to_plot.append('$\eta_{1}$')
        elif label == 'eta2':
            feature_names_list_to_plot.append('$\eta_{2}$')
        elif label == 'rapidity':
            feature_names_list_to_plot.append('$rapidity_{ll}$')
        else:
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

#print(feature_names_list_to_plot)

#scaler = StandardScaler().fit(trainX)
#trainX = scaler.transform(trainX)
#testX = scaler.transform(testX)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(trainX[0]),)),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='tanh'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='Adam', \
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
                  metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Train the model
history = model.fit(trainX, trainY, epochs= 80, validation_data=(testX, testY))#, batch_size=15)
#history_sm = model.fit(trainX_sm, trainY_sm, epochs= 30, validation_data=(testX_sm, testY_sm))

#history_df = pd.DataFrame.from_dict(history.history)

# FEATURE IMPORTANCE IMPLEMENTATION
# define the number of background samples to generate
num_background_samples = 100
# generate random input sequences that have the same shape as the training data3
input_background = np.random.normal(size=(num_background_samples,) + trainX.shape[1:])
# Create Shap explainer
explainer = shap.DeepExplainer(model, input_background)
# Calculate Shap values for all test samples
shap_values = explainer.shap_values(testX)

#print(plot_title, " --> Accuracy: {:.2f}%".format(history.history['accuracy'][-1]*100))

# Create and display Shap summary plot
shap.summary_plot(shap_values, testX, plot_type='bar', feature_names=feature_names_list_to_plot, show=False)


#history_sm = model.fit(trainX_sm, trainY_sm, epochs= 50, validation_data=(testX_sm, testY_sm))
fig, axis = plt.subplots(figsize =(20, 10))
axis.plot(history.history['accuracy'], label='Training Accuracy')
axis.plot(history.history['val_accuracy'], label='Testing Accuracy')
axis.set_xlabel('Epoch', fontsize = 35)
axis.set_ylabel('Accuracy', fontsize = 35)
plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
axis.legend(fontsize = 25)
#plt.title(plot_title + "\n Accuracy: {:.2f}%".format(history.history['accuracy'][-1]*100), fontsize=35)
plt.show()

print(plot_title + "\n Accuracy: {:.2f}%".format(history.history['accuracy'][-1]*100))
'''
    fig, axis = plt.subplots(1,2)#,figsize =(60, 30))
    axis[0].plot(history.history['accuracy'], label='Training Accuracy')
    axis[0].plot(history.history['val_accuracy'], label='Testing Accuracy')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Accuracy')
    axis[0].legend()
    axis[1].plot(history.history['loss'], label='Training Loss')
    axis[1].plot(history.history['val_loss'], label='Testing Loss')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Loss')
    axis[1].legend()
    #fig.suptitle("SMEFT \n" + plot_title )
    fig.tight_layout()
    plt.show()
    fig, axis = plt.subplots(1,2)#,figsize =(60, 30))
    axis[0].plot(history_sm.history['accuracy'], label='Training Accuracy')
    axis[0].plot(history_sm.history['val_accuracy'], label='Testing Accuracy')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Accuracy')
    axis[0].legend()
    axis[1].plot(history_sm.history['loss'], label='Training Loss')
    axis[1].plot(history_sm.history['val_loss'], label='Testing Loss')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Loss')
    axis[1].legend()
    #fig.suptitle("SM \n" + plot_title + "\n Nodes in hidden layer 1 = 128 \n Nodes in hidden layer 2 = 64")
    fig.tight_layout()
    plt.show()


# construct O_NN

def predict_prob(number):
  return [number[0],1-number[0]]
#HB
predictions = model.predict(testX)
for i in range(len(testY)):
    if predictions[i][0] != float(1.0000000e+00):
        predictions[i][0] = 0
        predictions[i][1] = float(1.0000000e+00)
    print(predictions[i], testY[i])

weight_prob = np.array(list(map(predict_prob, predictions)))
#p_plus = model.predict(testX)
print(weight_prob)
O_NN = []
for i in range(len(weight_prob)):
    O_NN.append(weight_prob[i][0]-weight_prob[i][1])
print(O_NN)

#sm
predictions_sm = model.predict(testX_sm)
for i in range(len(testY_sm)):
    if predictions_sm[i][0] != float(1.0000000e+00):
        predictions_sm[i][0] = 0
        predictions_sm[i][1] = float(1.0000000e+00)
    #print(predictions_sm[i], testY_sm[i])
weight_prob_sm = np.array(list(map(predict_prob, predictions_sm)))
#p_plus_sm = model.predict(testX_sm)
O_NN_sm = []
for i in range(len(weight_prob_sm)):
    O_NN_sm.append(weight_prob_sm[i][0]-weight_prob_sm[i][1])
print(O_NN_sm)

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
fig, axis = plt.subplots(figsize =(20, 10))
bins = 20

# HB = 0.1
n_err = np.sqrt(np.histogram(df[var], bins=bins, weights = (original_weights)**2)[0])
hist_values, bin_edges = np.histogram(df[var], bins, weights = original_weights)
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
axis.bar(bin_centres, 100*hist_values/bin_width, width=bin_width, alpha=0.2, color="red", label="HB = 0.1 * 100")
#axis.errorbar(bin_centres, (hist_values)*(-1), yerr=np.sqrt(100000), fmt='o', color="red", label = "HB")
# SM
n_err3 = np.sqrt(np.histogram(df2[var], bins=bins, weights = (CROSS_SECTION_PER_EVENT_sm*original_weights_sm)**2)[0])
hist_values3, bin_edges = np.histogram(df2[var], bins, weights = CROSS_SECTION_PER_EVENT_sm*original_weights_sm)
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
axis.errorbar(bin_centres, hist_values3/bin_width, yerr=n_err3, fmt='o', color="orange")
axis.bar(bin_centres, hist_values3/bin_width, width=bin_width, alpha=0.5, color="orange", label = "SM")

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
for i in range(len(trainX),len(x)):
    weights.append(original_weights.iloc[i])
    weights_squared.append((original_weights.iloc[i])**2)
    
weights_sm = []
weights_sm_squared = []
for i in range(len(trainX_sm),len(x_sm)):
    weights_sm.append(CROSS_SECTION_PER_EVENT_sm*original_weights_sm.iloc[i])
    weights_sm_squared.append((CROSS_SECTION_PER_EVENT_sm*original_weights_sm.iloc[i])**2)

bins = 2
# O_NN
n_err = np.sqrt(np.histogram(O_NN, bins=bins, weights = weights_squared)[0])
hist_values, bin_edges = np.histogram(O_NN, bins, weights = weights)
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
axis.bar(bin_centres, hist_values*1000/bin_width, width=bin_width, alpha=0.2, color="red", label="HB = 0.1*1000")
#axis.errorbar(bin_centres, (hist_values)*(-1), yerr=np.sqrt(100000), fmt='o', color="red", label = "HB")
# SM
n_err3 = np.sqrt(np.histogram(O_NN_sm, bins=bins, weights = weights_sm_squared)[0])
hist_values3, bin_edges = np.histogram(O_NN_sm, bins, weights = weights_sm)
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
axis.errorbar(bin_centres, hist_values3/bin_width, yerr=n_err3, fmt='o', color="orange")
axis.bar(bin_centres, hist_values3/bin_width, width=bin_width, alpha=0.5, color="orange", label = "SM")

plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
axis.set_xlabel('$O_{NN}$', fontsize = 35)
axis.set_ylabel("$\\frac{d\sigma}{d O_{NN}} [pb/O_{NN}]$", fontsize = 35)
#plt.title(, fontsize=35)
#axis.set_yscale('log', nonpositive='clip')
axis.legend(fontsize = 25)
axis.plot()
plt.show()

'''

