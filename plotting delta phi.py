import pandas as pd
import copy
import numpy as np
#import shap
import matplotlib.pyplot as plt

# files
filename = "eellh_HW_01.txt"
filename2 = "eellh_sm.txt"
filename3 = "eellh_HWB_01.txt"
filename4 = "eellh_HB_01.txt"

# Loading Data
df = pd.read_csv(filename, engine='python')
df2 = pd.read_csv(filename2,engine='python')
df3 = pd.read_csv(filename3, engine='python')
df4 = pd.read_csv(filename4, engine='python')
df.fillna(0, inplace=True)
df2.fillna(0, inplace=True)
df3.fillna(0, inplace=True)
df4.fillna(0, inplace=True)
df['event_weight'].value_counts()
feature_names = list(df.columns)
feature_names_sm = list(df2.columns)

original_weights_HW = copy.copy(df["event_weight"])
original_weights_sm = copy.copy(df2["event_weight"])
original_weights_HWB = copy.copy(df3["event_weight"])
original_weights_HB = copy.copy(df4["event_weight"])

df['delta_phi'] = np.where(df['delta_phi'] > np.pi, df['delta_phi']-2*np.pi, df['delta_phi'])
df['delta_phi'] = np.where(df['delta_phi'] < (-1)*np.pi, df['delta_phi']+2*np.pi, df['delta_phi'])
df2['delta_phi'] = np.where(df2['delta_phi'] > np.pi, df2['delta_phi']-2*np.pi, df2['delta_phi'])
df2['delta_phi'] = np.where(df2['delta_phi'] < (-1)*np.pi, df2['delta_phi']+2*np.pi, df2['delta_phi'])
df3['delta_phi'] = np.where(df3['delta_phi'] > np.pi, df3['delta_phi']-2*np.pi, df3['delta_phi'])
df3['delta_phi'] = np.where(df3['delta_phi'] < (-1)*np.pi, df3['delta_phi']+2*np.pi, df3['delta_phi'])
df4['delta_phi'] = np.where(df4['delta_phi'] > np.pi, df4['delta_phi']-2*np.pi, df4['delta_phi'])
df4['delta_phi'] = np.where(df4['delta_phi'] < (-1)*np.pi, df4['delta_phi']+2*np.pi, df4['delta_phi'])

# HW
WEIGHT_VALUE_HW = 0.000185707
SIGNAL_WEIGHTS_HW = df["sum_event_weights"].iloc[-1]
for i in range(len(original_weights_HW)):
    #print(original_weights.iloc[i])
    original_weights_HW.iloc[i] = original_weights_HW.iloc[i]/100000

#SM
TOTAL_CROSS_SECTION_sm = 1.484148e-02
#0.2383
TOTAL_WEIGHT_sm = 1484.148
#23830.2
SIGNAL_WEIGHTS_sm = df2["sum_event_weights"].iloc[-1]
CROSS_SECTION_PER_EVENT_sm = TOTAL_CROSS_SECTION_sm/TOTAL_WEIGHT_sm

# HWB
WEIGHT_VALUE_HWB = 0.00165399
SIGNAL_WEIGHTS_HWB = df3["sum_event_weights"].iloc[-1]
for i in range(len(original_weights_HWB)):
    #print(original_weights.iloc[i])
    original_weights_HWB.iloc[i] = original_weights_HWB.iloc[i]/100000
    
# HB
WEIGHT_VALUE_HB = 0.0129655
SIGNAL_WEIGHTS_HB = df4["sum_event_weights"].iloc[-1]
for i in range(len(original_weights_HB)):
    #print(original_weights.iloc[i])
    original_weights_HB.iloc[i] = original_weights_HB.iloc[i]/100000
    
var = "delta_phi"
LUMINOSITY = 5e6
fig, axis = plt.subplots(figsize =(20, 10))
bins = 20

#   PLOTTING DELTA PHI , width=bin_width, alpha=0.5

# SM
n_err2 = np.sqrt(np.histogram(df2[var], bins=bins, weights = (CROSS_SECTION_PER_EVENT_sm*original_weights_sm)**2)[0])
hist_values2, bin_edges = np.histogram(df2[var], bins, weights = CROSS_SECTION_PER_EVENT_sm*original_weights_sm)
n_deltaphi = np.histogram(df2[var], bins, weights = LUMINOSITY*CROSS_SECTION_PER_EVENT_sm*original_weights_sm)[0]
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
axis.errorbar(bin_centres, hist_values2/bin_width/10, yerr=n_err2/bin_width/10, fmt='o', color="darkorange")
first_hist_val = copy.copy(hist_values2[0])
new_hist_values2 = np.insert(hist_values2,0,first_hist_val,axis=None)
first_bin_val = copy.copy(bin_centres[0]-bin_width)
new_bin_centres = np.insert(bin_centres,0,first_bin_val,axis=None)
axis.step(new_bin_centres+bin_width/2, new_hist_values2/bin_width/10, color="darkorange", label = "SM/10")

# HW
n_err = np.sqrt(np.histogram(df[var], bins=bins, weights = (original_weights_HW)**2)[0])
hist_values, bin_edges = np.histogram(df[var], bins, weights = original_weights_HW)
EFT_deltaphi = np.histogram(df[var], bins, weights = LUMINOSITY*original_weights_HW)[0]
bin_width = (bin_edges[-1]-bin_edges[0])/bins
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
#axis.bar(bin_centres, 100*hist_values/bin_width, width=bin_width, alpha=0.2, color="blue", label=r"old")
axis.errorbar(bin_centres, 10*(hist_values)/bin_width, yerr=10*n_err/bin_width, fmt='o', color="crimson")
first_hist_val = copy.copy(hist_values[0])
new_hist_values = np.insert(hist_values,0,first_hist_val,axis=None)
first_bin_val = copy.copy(bin_centres[0]-bin_width)
new_bin_centres = np.insert(bin_centres,0,first_bin_val,axis=None)
axis.step(new_bin_centres+bin_width/2, 10*new_hist_values/bin_width, color="crimson", label=r"$c_{H\widetilde{W}}/\Lambda^{2}=1/TeV^{2}$")

# HWB
n_err = np.sqrt(np.histogram(df3[var], bins=bins, weights = (original_weights_HWB)**2)[0])
hist_values, bin_edges = np.histogram(df3[var], bins, weights = original_weights_HWB)
EFT_deltaphi = np.histogram(df3[var], bins, weights = LUMINOSITY*original_weights_HWB)[0]
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
#axis.bar(bin_centres, 100*hist_values/bin_width, width=bin_width, alpha=0.2, color="blue", label=r"old")
axis.errorbar(bin_centres, 5*(hist_values)/bin_width, yerr=5*n_err/bin_width, fmt='o', color="forestgreen")
first_hist_val = copy.copy(hist_values[0])
new_hist_values = np.insert(hist_values,0,first_hist_val,axis=None)
first_bin_val = copy.copy(bin_centres[0]-bin_width)
new_bin_centres = np.insert(bin_centres,0,first_bin_val,axis=None)
axis.step(new_bin_centres+bin_width/2, 5*new_hist_values/bin_width, color="forestgreen", label=r"$c_{H\widetilde{W}B}/\Lambda^{2}=0.5/TeV^{2}$")

# HW
n_err = np.sqrt(np.histogram(df4[var], bins=bins, weights = (original_weights_HB)**2)[0])
hist_values, bin_edges = np.histogram(df4[var], bins, weights = original_weights_HB)
EFT_deltaphi = np.histogram(df4[var], bins, weights = LUMINOSITY*original_weights_HB)[0]
bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_width = (bin_edges[-1]-bin_edges[0])/bins
#axis.bar(bin_centres, 100*hist_values/bin_width, width=bin_width, alpha=0.2, color="blue", label=r"old")
axis.errorbar(bin_centres, 10*(hist_values)/bin_width, yerr=10*n_err/bin_width, fmt='o', color="purple")
first_hist_val = copy.copy(hist_values[0])
new_hist_values = np.insert(hist_values,0,first_hist_val,axis=None)
first_bin_val = copy.copy(bin_centres[0]-bin_width)
new_bin_centres = np.insert(bin_centres,0,first_bin_val,axis=None)
axis.step(new_bin_centres+bin_width/2, 10*new_hist_values/bin_width, color="purple", label=r"$c_{H\widetilde{B}}/\Lambda^{2}=1/TeV^{2}$")

plt.axhline(0,color='lightgray')

plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
axis.set_xlabel('$\Delta \phi$[rad]', fontsize = 35)
axis.set_ylabel("$\\frac{d\sigma}{d \Delta \phi}$ [pb/rad]", fontsize = 35)
#plt.title(, fontsize=35)
#axis.set_yscale('log', nonpositive='clip')
axis.legend(fontsize = 25)
axis.plot()
plt.show()
