import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps
from sklearn.svm import SVR, NuSVR, NuSVC
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os
import shutil
import posixpath

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.append("/home/pdp1145/fetal_ecg_det/wfdb_python_master/")

# import kymatio

import wfdb
import pywt

# record = wfdb.rdrecord('/home/pdp1145/fetal_ecg_det/wfdb_python_master/sample-data/a103l')
record = wfdb.rdrecord('/home/pdp1145/fetal_ecg_det/fetal_ecg_data/ARR_01')
# record = wfdb.rdrecord('/home/pdp1145/fetal_ecg_det/fetal_ecg_data/NR_06')
# wfdb.plot_wfdb(record=record, title='Record a103l from Physionet Challenge 2015')

mat_lead = record.p_signal[0:15000,0]
fetal_lead = record.p_signal[0:15000,1]

x = np.arange(len(mat_lead))

fig = go.Figure(data=go.Scatter(x=x, y=fetal_lead))
fig = make_subplots(rows=2, cols=1)
fig.append_trace(go.Scatter(x=x, y=mat_lead), row=1, col=1)
fig.append_trace(go.Scatter(x=x, y=fetal_lead), row=2, col=1)
fig.show()

widths = np.arange(1, 129)*0.75
cwt_maternal_lead = sps.cwt(mat_lead, sps.ricker, widths)
# fig = px.imshow(cwt_maternal_lead)
# fig.add_layout_image(dict(sizing="stretch"))
# fig.show()
plt.imshow(cwt_maternal_lead, aspect='auto')   # , extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

cwt_trans = np.transpose(cwt_maternal_lead)

plt.imshow(cwt_trans)

# SVR w/ single CWT vector -> fetal ECG:
#

wdw_lth_h = 2
n_feats = wdw_lth_h*2*128
wdw_beg = 1000
wdw_end = 2000
regr_idx = 0
fetal_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
mat_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
cwt_wdw = np.zeros([(wdw_end - wdw_beg), n_feats])

for wdw_idx in np.arange(wdw_beg, wdw_end):
    fetal_lead_wdw[regr_idx] = fetal_lead[wdw_idx]
    mat_lead_wdw[regr_idx] = mat_lead[wdw_idx]
    blef = cwt_trans[wdw_idx - wdw_lth_h : wdw_idx + wdw_lth_h, :]
    cwt_wdw[regr_idx,:] = blef.flatten()
    regr_idx = regr_idx +1

# reg = SGDRegressor(max_iter=10000, n_iter_no_change = 10, learning_rate = 'constant', early_stopping=True)
# # reg.fit(cwt_trans[500:1000,:], fetal_lead[500:1000])
# reg.fit(cwt_wdw, fetal_lead_wdw)
# mat_pred=reg.predict(cwt_wdw)
#
# # plt.plot(fetal_lead, reg.predict(cwt_trans))
# # plt.show()
#
# # figx = go.Figure(data=go.Scatter(x=x, y=fetal_lead))
# figx = make_subplots(rows=2, cols=1)
# figx.append_trace(go.Scatter(x=x, y=fetal_lead_wdw), row=1, col=1)
# figx.append_trace(go.Scatter(x=x, y=mat_pred), row=2, col=1)
# figx.show()

# plt.plot(fetal_lead[500:700])
# plt.plot(reg.predict(cwt_trans[500:700,:]))
#

x_idxs = np.arange(len(fetal_lead))

# oresvr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.0001)
# y_rbf = oresvr_rbf.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)
# # y_rbf = svr_rbf.fit(cwt_trans[500:1000,:], fetal_lead[500:1000]).predict(cwt_trans[500:1000,:])
#
# figy = make_subplots(rows=2, cols=1, subplot_titles=("Maternal", "Fetal & RBF SVR Estimate"))
# figy.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
# figy.append_trace(go.Scatter(x = x_idxs, y = fetal_lead_wdw), row=2, col=1)
# figy.append_trace(go.Scatter(x = x_idxs, y = y_rbf), row=2, col=1)
# figy.show()


nusv_res = NuSVR(nu=0.75, C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
z_rbf = nusv_res.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)
# z_rbf = nusv_res.fit(cwt_wdw, mat_lead_wdw).predict(cwt_wdw)
nusv_lin_coef = nusv_res.coef_
cwt_wdw_trans = np.transpose(cwt_wdw)
z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
z_cwt_xcoef_rs = np.reshape(z_cwt_xcoef, (1000,))

figz = make_subplots(rows=2, cols=1, subplot_titles=("Maternal", "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4, Training Record Length = 5000"))
figz.append_trace(go.Scatter(x = x_idxs, y = z_cwt_xcoef_rs), row=1, col=1)
figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=1, col=1)
figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=2, col=1)
figz.show()

# figz = make_subplots(rows=2, cols=1, subplot_titles=("Maternal", "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4, Training Record Length = 5000"))
# figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
# figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=2, col=1)
# figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=2, col=1)
# figz.show()


# Run trained SVR on full record:
#
wdw_beg = 1
wdw_end = 15000
regr_idx = 0
fetal_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
mat_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
cwt_wdw = np.zeros([(wdw_end - wdw_beg), n_feats])
for wdw_idx in np.arange(wdw_beg, wdw_end):
    fetal_lead_wdw[regr_idx] = fetal_lead[wdw_idx]
    mat_lead_wdw[regr_idx] = mat_lead[wdw_idx]
    blef = cwt_trans[wdw_idx - wdw_lth_h : wdw_idx + wdw_lth_h, :]
    cwt_wdw[regr_idx,:] = blef.flatten()
    regr_idx = regr_idx +1

z_rbf = nusv_res.predict(cwt_wdw)
figz = make_subplots(rows=2, cols=1)
figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
figz.append_trace(go.Scatter(x = x_idxs, y = fetal_lead_wdw), row=2, col=1)
figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=2, col=1)
figz.show()


# plt.plot(fetal_lead[500:700])
# plt.plot(svr_rbf.predict(cwt_trans[500:700,:]))

arf = 12



