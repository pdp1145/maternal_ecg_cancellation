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
import time

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
# figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal", "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4"))

cwt_wdw_lth_h = 2
n_feats = cwt_wdw_lth_h*2*128
svr_wdw_lth = 120
n_coef_tpls = 1000
init_delay = 1000

for svr_wdw_beg in np.arange(init_delay, init_delay + n_coef_tpls, 20):

    wdw_beg = svr_wdw_beg
    wdw_end = wdw_beg + svr_wdw_lth
    fetal_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
    mat_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
    cwt_wdw = np.zeros([(wdw_end - wdw_beg), n_feats])
    regr_idx = 0

    for wdw_idx in np.arange(wdw_beg, wdw_end):
        fetal_lead_wdw[regr_idx] = fetal_lead[wdw_idx]
        mat_lead_wdw[regr_idx] = mat_lead[wdw_idx]
        blef = cwt_trans[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h, :]
        cwt_wdw[regr_idx,:] = blef.flatten()
        regr_idx = regr_idx +1

    x_idxs = np.arange(len(fetal_lead))

    nusv_res = NuSVR(nu=0.75, C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
    z_rbf = nusv_res.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)
    # z_rbf = nusv_res.fit(cwt_wdw, mat_lead_wdw).predict(cwt_wdw)
    nusv_lin_coef = nusv_res.coef_
    cwt_wdw_trans = np.transpose(cwt_wdw)
    z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
    z_cwt_xcoef_rs = np.reshape(z_cwt_xcoef, (svr_wdw_lth,)) + nusv_res.intercept_

    fig_mpl, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    ax0.plot(mat_lead_wdw)
    ax0.set_title('Maternal')
    ax1.plot(fetal_lead_wdw)
    ax1.set_title('Abdominal')
    ax2.plot(z_cwt_xcoef_rs)
    ax2.plot(z_rbf)
    ax2.set_title('SVR Estimate')
    mngr = plt.get_current_fig_manager()
    mngr.full_screen_toggle()
    fig_mpl.show()
    time.sleep(5)
    plt.close(fig_mpl)

    # figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
    # figz.append_trace(go.Scatter(x = x_idxs, y = fetal_lead_wdw), row=2, col=1)
    # figz.append_trace(go.Scatter(x = x_idxs, y = z_cwt_xcoef_rs), row=3, col=1)
    # figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=3, col=1)
    # figz.show()
    # figz.data = []

    arf = 12
# figz = make_subplots(rows=2, cols=1, subplot_titles=("Maternal", "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4, Training Record Length = 5000"))
# figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
# figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=2, col=1)
# figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=2, col=1)
# figz.show()

# matplotlib.pyplot.close()

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
    blef = cwt_trans[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h, :]
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



