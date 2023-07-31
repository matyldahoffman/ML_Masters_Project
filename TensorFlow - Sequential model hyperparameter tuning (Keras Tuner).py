import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#import xgboost as xgb
#from sklearn.metrics import accuracy_score,confusion_matrix
#import shap
#shap.initjs()
#import seaborn as sns
#import os
import keras_tuner as kt

def model_builder(hp): # hp passes in hyperparameters()
    model = tf.keras.Sequential()
    #model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    #hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    model.add(tf.keras.layers.Dense(units=hp.Int('Nodes in entry layer', \
                                                 min_value=64, max_value=128, step=32),\
                                    activation='relu', input_shape=(len(trainX[0]),)))
    for i in range(hp.Int('num_layers',1,3)):
        model.add(tf.keras.layers.Dense(units=hp.Int('nodes_' + str(i), \
                                                     min_value=64, max_value=128, step=32), \
                                        activation=hp.Choice('dense_activation_' + str(i), \
                                                             ['relu', 'tanh', 'sigmoid', 'linear'])))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.2, step=0.1)))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    #hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),\
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

# files
filename = "eellh_HB_01.txt"

# Loading Data
df = pd.read_csv(filename,engine='python').dropna()
df['event_weight'].value_counts()

var_list = ['event_weight','sum_event_weights','n_photon','n_electron','n_muon',\
            'rapidity1','E1','Et1','px1','py1','pt1','mass1',\
                'pt2','rapidity2','E2','Et2','px2','py2','mass2',\
                    'eta','pt','phi','rapidity','E','Et','px','py','mass','m_recoil','ang_sep','delta_phi',\
                        'phi1','phi2','eta1','eta2']
var_drop = ['event_weight','sum_event_weights','n_photon','n_electron','n_muon',\
            'rapidity1','E1','Et1','px1','py1','pt1','mass1',\
                'pt2','rapidity2','E2','Et2','px2','py2','mass2',\
                    'eta','pt','phi','E','Et','px','py','mass','m_recoil','ang_sep','delta_phi']
    
plot_title = 'Training variables = '

df['delta_phi'] = np.where(df['delta_phi'] > np.pi, df['delta_phi']-2*np.pi, df['delta_phi'])
df['delta_phi'] = np.where(df['delta_phi'] < (-1)*np.pi, df['delta_phi']+2*np.pi, df['delta_phi'])

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

# axis = 1 tells us to drop labels from the columns and not just the index (that would be x=0)
y = np.asarray(df['event_weight'])
# 
# E, px, py
# normalize_value = (value − min_value) / (max_value − min_value)

#x = tf.truediv(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))

# Data pre-processing
trainX, testX = x[:int(0.8*len(x))], x[int(0.8*len(x)):]
trainY, testY = y[:int(0.8*len(y))], y[int(0.8*len(y)):]
# Normalise the event_weights = Y arrays
trainY[np.where(trainY < 0)] = 0
trainY[np.where(trainY > 0)] = 1
testY[np.where(testY < 0)] = 0
testY[np.where(testY > 0)] = 1

#scaler = StandardScaler().fit(trainX)

#trainX = scaler.transform(trainX)
#testX = scaler.transform(testX)

print(trainX)
print(trainY)

# using the hyperband tuner (build a tuner object)
tuner = kt.Hyperband(model_builder,
    objective='val_accuracy',
    max_epochs=100,
    factor=3,
    directory='my_dir',
    project_name='tune_hyperparameters',
    overwrite=True)

# stop early if the validation loss does not improve for 5 epochs 
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner.search(trainX, trainY, epochs=100, validation_data=(testX, testY), callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the optimal values for the hyperparameters
print(f"Optimal number of nodes in entry layer: {best_hps.get('Nodes in entry layer')}")
print(f"Optimal number of layers: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f"Optimal number of nodes in layer {i}: {best_hps.get('nodes_' + str(i))}")
    print(f"Optimal activation function in layer {i}: {best_hps.get('dense_activation_' + str(i))}")
    print(f"Optimal dropout rate for layer {i}: {best_hps.get('dropout_' + str(i))}")

# Find the optimal number of epochs to trani the model with the hyperparameters obtained from the search
model = tuner.hypermodel.build(best_hps)
history = model.fit(trainX, trainY, epochs=100, validation_data=(testX, testY))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#Re-instantiate the hypermodel and train it with the optimal number of epochs from above
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(trainX, trainY, epochs=best_epoch, validation_data=(testX, testY))


'''
# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(trainX[0]),)),
    #tf.keras.layers.Dropout(0.8,activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Compile the model
model.compile(optimizer='Adam', \
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
                  metrics=['accuracy'])
'''
# Train the model
#history = model.fit(trainX, trainY, epochs= 10, validation_data=(testX, testY))#, batch_size=15)
history_df = pd.DataFrame.from_dict(history.history)
#sns.lineplot(data=history_df[['loss', 'val_loss']])

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title(plot_title)
plt.show()

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
fig.suptitle(plot_title)
fig.tight_layout()
plt.show()

best_hps.values

'''

#explainer = shap.Explainer(model)
#shap_values = explainer(trainX)

explainer = shap.KernelExplainer(model, trainX[:50,:])
shap_values = explainer.shap_values(trainX[20,:], nsamples=500)
plt.switch_backend('agg')
shap.summary_plot(shap_values, trainX, max_display=trainX.shape[1])
#shap.plots.waterfall(shap_values)
#shap.plots.force(shap_values)

#shap.force_plot(explainer.expected_value[0], shap_values[0], trainX[20,:], matplotlib=True)

plt.savefig('summary_plot_result.jpg')

#shap_values50 = explainer.shap_values(trainX[50:100,:], nsamples=500)
#shap_display2 = shap.force_plot(explainer.expected_value[0], shap_values50[0], trainX[50:100,:], matplotlib=True)
'''
