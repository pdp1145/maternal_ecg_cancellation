import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps
from sklearn.svm import SVR, NuSVR, NuSVC
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os
import shutil
import posixpath
import time
# from pynput import mouse

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
cwt_fetal_lead = sps.cwt(mat_lead, sps.ricker, widths)

cwt_trans = np.transpose(cwt_maternal_lead)
cwt_trans_fetal = np.transpose(cwt_fetal_lead)

fig_cwt_mat, ax_cwt_mat = plt.subplots()
ax_cwt_mat.imshow(cwt_maternal_lead, aspect='auto')

fig_cwt_fetal, ax_cwt_fetal = plt.subplots()
ax_cwt_fetal.imshow(cwt_fetal_lead, aspect='auto')




# SVR w/ single CWT vector -> fetal ECG:
#
# figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal", "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4"))

cwt_wdw_lth_h = 8
n_feats = cwt_wdw_lth_h*2*128
svr_wdw_lth = 32
n_feats_maternal = n_feats*svr_wdw_lth
n_feats_maternal_fetal = n_feats*2*svr_wdw_lth

n_coef_tpls = 10000
maternal_feature_vectors = np.zeros([n_coef_tpls, n_feats_maternal])
maternal_fetal_feature_vectors = np.zeros([n_coef_tpls, n_feats_maternal_fetal])
linear_regression_coefs = np.zeros([n_coef_tpls, n_feats])
linear_regression_intercepts = np.zeros([n_coef_tpls,])

n_maternal_fetal_feature_vectors = 0
init_delay = 1000
wdw_shift = 1

# fig_mpl, (ax0, ax1, ax2) = plt.subplots(nrows=3)

abdominal_est = np.zeros(n_coef_tpls,)
abdominal_est_idxs = np.arange(0, n_coef_tpls)
n_svrs = 0
overlap_wdw_idx = 0

for svr_wdw_beg in np.arange(init_delay, init_delay + n_coef_tpls, wdw_shift):

    wdw_beg = svr_wdw_beg
    wdw_end = wdw_beg + svr_wdw_lth
    fetal_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
    mat_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
    cwt_wdw = np.zeros([(wdw_end - wdw_beg), n_feats])
    cwt_wdw_fetal = np.zeros([(wdw_end - wdw_beg), n_feats])
    regr_idx = 0

    for wdw_idx in np.arange(wdw_beg, wdw_end):

        fetal_lead_wdw[regr_idx] = fetal_lead[wdw_idx]    # Extract lead windows for regression
        mat_lead_wdw[regr_idx] = mat_lead[wdw_idx]

        # blef = cwt_trans[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h, :]     # Extract feature vectors for regression & knn
        # cwt_wdw[regr_idx,:] = blef.flatten()

        cwt_wdw[regr_idx,:] = cwt_trans[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h, :].flatten()     # Extract feature vectors for regression & knn
        cwt_wdw_fetal[regr_idx,:] = cwt_trans[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h, :].flatten()     # Extract feature vectors for regression & knn

        regr_idx = regr_idx +1

    maternal_feature_vectors[n_svrs, :] = cwt_wdw.flatten()
    maternal_fetal_feature_vectors[n_svrs, :] = np.concatenate((cwt_wdw.flatten(), cwt_wdw_fetal.flatten()), axis = None)
    x_idxs = np.arange(len(fetal_lead))

    nusv_res = NuSVR(nu=0.75, C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
    z_rbf = nusv_res.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)
    # z_rbf = nusv_res.fit(cwt_wdw, mat_lead_wdw).predict(cwt_wdw)
    
    nusv_lin_coef = nusv_res.coef_
    cwt_wdw_trans = np.transpose(cwt_wdw)
    z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
    z_cwt_xcoef_rs = np.reshape(z_cwt_xcoef, (svr_wdw_lth,)) + nusv_res.intercept_

    linear_regression_coefs[n_svrs, :] = nusv_lin_coef
    linear_regression_intercepts[n_svrs] = nusv_res.intercept_

    abdominal_est[overlap_wdw_idx : (overlap_wdw_idx + svr_wdw_lth)] = np.add(z_cwt_xcoef_rs, abdominal_est[overlap_wdw_idx : (overlap_wdw_idx + svr_wdw_lth)])
    overlap_wdw_idx = overlap_wdw_idx +1
    n_svrs = n_svrs +1

    # plt.close(fig_mpl)
    # fig_mpl, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    # ax0.plot(mat_lead_wdw)
    # ax0.set_title('Maternal')
    # ax1.plot(fetal_lead_wdw)
    # ax1.set_title('Abdominal')
    # ax2.plot(z_cwt_xcoef_rs)
    # ax2.plot(z_rbf)
    # ax2.set_title('SVR Estimate')
    # mngr = plt.get_current_fig_manager()
    # mngr.full_screen_toggle()
    # fig_mpl.show()

    if((n_svrs % 50) == 1214):
        figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal",
                "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4, Training Record Length = 5000", "Abdominal Estimate"))
        figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
        figz.append_trace(go.Scatter(x = x_idxs, y = fetal_lead_wdw), row=2, col=1)
        figz.append_trace(go.Scatter(x = x_idxs, y = z_cwt_xcoef_rs), row=3, col=1)
        figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=3, col=1)
        figz.show()   
        time.sleep(5.0)

    if ((n_svrs % 1000) == 0):
        figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal", "Abdominal Estimate"))
        figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead[init_delay : (init_delay + n_coef_tpls)]), row=1, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead[init_delay : (init_delay + n_coef_tpls)]), row=2, col=1)
        figz.append_trace(go.Scatter(x=abdominal_est_idxs, y=abdominal_est), row=3, col=1)
        figz.show()
        time.sleep(5.0)
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



