import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps
from sklearn.svm import SVR, NuSVR, NuSVC
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from scipy.spatial import distance_matrix
from scipy.spatial import distance

# from sklearn.neighbors import DistanceMetric
# from sklearn.cluster import KMeans

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os
import shutil
import posixpath
import time
from timeit import default_timer as timer

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

rem_record_lth = 25000

mat_lead = np.float32(record.p_signal[0:rem_record_lth,0])
fetal_lead = np.float32(record.p_signal[0:rem_record_lth,1])

x = np.arange(len(mat_lead))

fig = go.Figure(data=go.Scatter(x=x, y=fetal_lead))
fig = make_subplots(rows=2, cols=1)
fig.append_trace(go.Scatter(x=x, y=mat_lead), row=1, col=1)
fig.append_trace(go.Scatter(x=x, y=fetal_lead), row=2, col=1)
fig.show()

widths = np.arange(1, 129)*0.75
cwt_maternal_lead = np.float32(sps.cwt(mat_lead, sps.ricker, widths))
cwt_fetal_lead = np.float32(sps.cwt(fetal_lead, sps.ricker, widths))

cwt_trans = np.float32(np.transpose(cwt_maternal_lead))
cwt_trans_fetal = np.float32(np.transpose(cwt_fetal_lead))

fig_cwt_mat, ax_cwt_mat = plt.subplots()
ax_cwt_mat.imshow(cwt_maternal_lead[:, 0 : 1000], aspect='auto')
time.sleep(5.0)
fig_cwt_fetal, ax_cwt_fetal = plt.subplots()
ax_cwt_fetal.imshow(cwt_fetal_lead[:, 0 : 1000], aspect='auto')
time.sleep(5.0)

# SVR w/ single CWT vector -> fetal ECG:
#
# figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal", "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4"))

cwt_wdw_lth_h = 30
n_feats = cwt_wdw_lth_h*2*128
svr_wdw_lth = 64
n_feats_maternal = n_feats*svr_wdw_lth
n_feats_maternal_fetal = n_feats*2*svr_wdw_lth

n_coef_tpls = 1000
maternal_feature_vectors = np.float32(np.zeros([n_coef_tpls, n_feats_maternal]))
maternal_fetal_feature_vectors = np.float32(np.zeros([n_coef_tpls, n_feats_maternal_fetal]))
linear_regression_coefs = np.float32(np.zeros([n_coef_tpls, n_feats]))
linear_regression_intercepts = np.float32(np.zeros([n_coef_tpls,]))

n_maternal_fetal_feature_vectors = 0
init_record_skip = 1000
init_delay = init_record_skip
wdw_shift = 1

# fig_mpl, (ax0, ax1, ax2) = plt.subplots(nrows=3)

abdominal_est = np.float32(np.zeros(rem_record_lth,))
abdominal_est_idxs = np.arange(0, n_coef_tpls)
dist_arr = np.float32(np.zeros(rem_record_lth,))
n_svrs = 0
overlap_wdw_idx = 0
init = 0

if(init == 1):  # Load initialized template library and regressors if already initialized
    maternal_fetal_feature_vectors = np.load('maternal_fetal_feature_vectors1k.npy')
    maternal_feature_vectors = np.float32(np.load('maternal_feature_vectors1k.npy'))
    linear_regression_coefs = np.float32(np.load('linear_regression_coefs1k.npy'))
    linear_regression_intercepts = np.float32(np.load('linear_regression_intercepts1k.npy'))

    n_svrs = n_coef_tpls                # Skip past template library initialization
    init_delay = init_delay + n_svrs
    overlap_wdw_idx = overlap_wdw_idx + n_svrs


for svr_wdw_beg in np.arange(init_delay, init_delay + rem_record_lth - svr_wdw_lth -1, wdw_shift):

        init_sect_beg = timer()

        wdw_beg = svr_wdw_beg
        wdw_end = wdw_beg + svr_wdw_lth
        fetal_lead_wdw = np.float32(np.zeros([(wdw_end - wdw_beg),]))
        mat_lead_wdw = np.float32(np.zeros([(wdw_end - wdw_beg),]))
        cwt_wdw = np.float32(np.zeros([(wdw_end - wdw_beg), n_feats]))
        cwt_wdw_fetal = np.float32(np.zeros([(wdw_end - wdw_beg), n_feats]))
        regr_idx = 0
        init_sect_end = timer()
        # print(" Init sect elapsed time:  @  " + str(svr_wdw_beg) + "      "   +  str(init_sect_end - init_sect_beg))
        init_sect_beg = timer()

        for wdw_idx in np.arange(wdw_beg, wdw_end):

            fetal_lead_wdw[regr_idx] = fetal_lead[wdw_idx]    # Extract lead windows for regression
            mat_lead_wdw[regr_idx] = mat_lead[wdw_idx]

            cwt_wdw[regr_idx,:] = cwt_trans[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h, :].flatten()     # Extract feature vectors for regression & knn
            cwt_wdw_fetal[regr_idx,:] = cwt_trans[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h, :].flatten()     # Extract feature vectors for regression & knn

            regr_idx = regr_idx +1
            
        init_sect_end = timer()
        # print(" Array collection sect elapsed time:  @  " + str(svr_wdw_beg) + "      "   +  str(init_sect_end - init_sect_beg))

        if(n_svrs < n_coef_tpls):       # Initialization phase (fill template library)

            init_sect_beg = timer()

            maternal_feature_vectors[n_svrs, :] = cwt_wdw.flatten()
            maternal_fetal_feature_vectors[n_svrs, :] = np.concatenate((cwt_wdw.flatten(), cwt_wdw_fetal.flatten()), axis = None)

            nusv_res = NuSVR(nu=0.95, C=10.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=10000)
            z_rbf = nusv_res.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)
            # z_rbf = nusv_res.fit(cwt_wdw, mat_lead_wdw).predict(cwt_wdw)

            nusv_lin_coef = np.float32(nusv_res.coef_)

            linear_regression_coefs[n_svrs, :] = np.float32(nusv_lin_coef)
            linear_regression_intercepts[n_svrs] = np.float32(nusv_res.intercept_)
            nusv_intercept = np.float32(nusv_res.intercept_)
            
            init_sect_end = timer()
            # print(" NuSVR sect elapsed time:  @  " + str(svr_wdw_beg) + "      " + str(init_sect_end - init_sect_beg))

        else:       # Estimates based on retrieved templates / update templates

            # Maternal & fetal CWT templates centered on this sample:
            maternal_feature_vector_s = cwt_wdw.flatten()
            maternal_feature_vector_rs = np.reshape(maternal_feature_vector_s, (1, maternal_feature_vector_s.size))

            maternal_fetal_feature_vector_s = np.concatenate((cwt_wdw.flatten(), cwt_wdw_fetal.flatten()), axis=None)

            # Get k-nn maternal lead templates:
            token_dists_knn = distance.cdist(maternal_feature_vector_rs, maternal_feature_vectors, metric='cityblock')

            token_dists_knn_sorted_idxs = np.argsort(token_dists_knn).flatten()
            token_dists_knn_fl = token_dists_knn.flatten()
            token_dists_knn_sorted = token_dists_knn_fl[token_dists_knn_sorted_idxs]
            
            dist_arr[init_record_skip + overlap_wdw_idx + int(svr_wdw_lth / 2)] = token_dists_knn_sorted[0]

            #
            # token_dist_knn_idxs = np.arange(len(token_dists_knn_sorted))
            # fig = make_subplots(rows=1, cols=1)
            # fig.append_trace(go.Scatter(x=token_dist_knn_idxs, y=token_dists_knn_sorted), row=1, col=1)
            # fig.show()

            # Retrieve regression coef's from best matches:
            #
            nusv_lin_coef = linear_regression_coefs[token_dists_knn_sorted_idxs[0], :]
            nusv_intercept = linear_regression_intercepts[token_dists_knn_sorted_idxs[0]]

            # # Generate abdominal signal estimates:
            # #
            # cwt_wdw_trans = np.transpose(cwt_wdw)
            # z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
            # z_cwt_xcoef_rs = z_cwt_xcoef + nusv_intercept

        # Generate abdominal signal estimate for this window:
        #
        cwt_wdw_trans = np.transpose(cwt_wdw)
        z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
        z_cwt_xcoef_rs = np.reshape(z_cwt_xcoef, (svr_wdw_lth,)) + nusv_intercept

#        abdominal_est[(init_record_skip + overlap_wdw_idx) : (init_record_skip + overlap_wdw_idx + svr_wdw_lth)] = \
#                               np.add(z_cwt_xcoef_rs, abdominal_est[(init_record_skip + overlap_wdw_idx) : (init_record_skip + overlap_wdw_idx + svr_wdw_lth)])
        abdominal_est[init_record_skip + overlap_wdw_idx + int(svr_wdw_lth/2)] = \
                                np.add(z_cwt_xcoef_rs[int(svr_wdw_lth/2)], abdominal_est[init_record_skip + overlap_wdw_idx + int(svr_wdw_lth/2)])

        overlap_wdw_idx = overlap_wdw_idx +1
        n_svrs = n_svrs +1


        if((n_svrs % 50) == 1214):
            figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal",
                    "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4, Training Record Length = 5000", "Abdominal Estimate"))
            figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
            figz.append_trace(go.Scatter(x = x_idxs, y = fetal_lead_wdw), row=2, col=1)
            figz.append_trace(go.Scatter(x = x_idxs, y = z_cwt_xcoef_rs), row=3, col=1)
            figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=3, col=1)
            figz.show()
            time.sleep(5.0)

        if ((n_svrs % 400) == 0):
            x_idxs = np.arange(init_record_skip, (init_record_skip + overlap_wdw_idx))
            figz = make_subplots(rows=4, cols=1, subplot_titles=("Maternal", "Abdominal", "Abdominal Estimate"))
            figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead[init_record_skip : (init_record_skip + overlap_wdw_idx)]), row=1, col=1)
            figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead[init_record_skip : (init_record_skip + overlap_wdw_idx)]), row=2, col=1)
            figz.append_trace(go.Scatter(x=x_idxs, y=abdominal_est[init_record_skip : (init_record_skip + overlap_wdw_idx)]), row=3, col=1)
            figz.append_trace(go.Scatter(x=x_idxs, y=dist_arr[init_record_skip : (init_record_skip + overlap_wdw_idx)]), row=4, col=1)
            figz.show()
            time.sleep(10.0)

            if(init == 0):
                np.save('maternal_fetal_feature_vectors1k', maternal_fetal_feature_vectors, allow_pickle=False)
                np.save('maternal_feature_vectors1k', maternal_feature_vectors, allow_pickle=False)
                np.save('linear_regression_coefs1k', linear_regression_coefs, allow_pickle=False)
                np.save('linear_regression_intercepts1k', linear_regression_intercepts, allow_pickle=False)
            # figz.data = []

        if ((n_svrs % 25) == 0):
            print(['n_svrs:  ' + str(n_svrs)])


# Get histogram of token - token distances for clustering:
#
dist = DistanceMetric.get_metric('manhattan')
token_dists = dist.pairwise(maternal_fetal_feature_vectors[0:200,:])

# token_dists = distance_matrix(maternal_fetal_feature_vectors, maternal_fetal_feature_vectors, p=1, threshold=100000000)
# token_dists = distance.cdist(maternal_fetal_feature_vectors[0:5,:], maternal_fetal_feature_vectors[0:5,:], metric='cityblock')
token_dists = distance.pdist(maternal_fetal_feature_vectors, metric='cityblock')

# token_dist_hist = np.histogram(token_dists, bins=1000)
# token_dist_hist_idxs = np.arange(len(token_dist_hist))

token_dists_sorted = np.sort(token_dists)
token_dist_idxs = np.arange(len(token_dists_sorted))
fig = make_subplots(rows=1, cols=1)
fig.append_trace(go.Scatter(x=token_dist_idxs, y=token_dists_sorted), row=1, col=1)
fig.show()

# kmeans_maternal_fetal = KMeans(n_clusters = 100, init = 'k-means++').fit(maternal_fetal_feature_vectors)

post_init = init_delay + n_coef_tpls - svr_wdw_lth   # Post-init processing w/ no gap

for svr_wdw_beg in np.arange(post_init, post_init + rem_record_lth - svr_wdw_lth, wdw_shift):

    wdw_beg = svr_wdw_beg
    wdw_end = wdw_beg + svr_wdw_lth
    fetal_lead_wdw = np.float32(np.zeros([(wdw_end - wdw_beg), ]))
    mat_lead_wdw = np.float32(np.zeros([(wdw_end - wdw_beg), ]))
    cwt_wdw = np.float32(np.zeros([(wdw_end - wdw_beg), n_feats]))
    cwt_wdw_fetal = np.float32(np.zeros([(wdw_end - wdw_beg), n_feats]))
    regr_idx = 0

    # Snapshot of maternal and fetal CWT contexts (templates) and CWT feature vectors for regression
    for wdw_idx in np.arange(wdw_beg, wdw_end):
        fetal_lead_wdw[regr_idx] = fetal_lead[wdw_idx]  # Extract lead windows for regression
        mat_lead_wdw[regr_idx] = mat_lead[wdw_idx]

        cwt_wdw[regr_idx, :] = cwt_trans[wdw_idx - cwt_wdw_lth_h: wdw_idx + cwt_wdw_lth_h,:].flatten()  # Extract feature vectors for regression & knn
        cwt_wdw_fetal[regr_idx, :] = cwt_trans[wdw_idx - cwt_wdw_lth_h: wdw_idx + cwt_wdw_lth_h,:].flatten()  # Extract feature vectors for regression & knn

        regr_idx = regr_idx + 1

    # Maternal & fetal CWT templates centered on this sample:
    maternal_feature_vector_s = cwt_wdw.flatten()
    maternal_feature_vector_rs = np.reshape(maternal_feature_vector_s, (1, maternal_feature_vector_s.size))

    maternal_fetal_feature_vector_s = np.concatenate((cwt_wdw.flatten(), cwt_wdw_fetal.flatten()), axis=None)

    # Get k-nn maternal lead templates:
    token_dists_knn = distance.cdist(maternal_feature_vector_rs, maternal_feature_vectors, metric='cityblock')
    token_dists_knn_fl = token_dists_knn.flatten()

    token_dists_knn_sorted_idxs = np.argsort(token_dists_knn).flatten()
    token_dists_knn_fl = token_dists_knn.flatten()
    token_dists_knn_sorted = token_dists_knn_fl[token_dists_knn_sorted_idxs]
    token_dist_knn_idxs = np.arange(len(token_dists_knn_sorted))

    #
    # fig = make_subplots(rows=1, cols=1)
    # fig.append_trace(go.Scatter(x=token_dist_knn_idxs, y=token_dists_knn_sorted), row=1, col=1)
    # fig.show()

    # Retrieve regression coef's from best matches:
    #
    nusv_lin_coef = linear_regression_coefs[token_dists_knn_sorted_idxs[0], :]
    nusv_intercept = linear_regression_intercepts[token_dists_knn_sorted_idxs[0]]
    
    # Generate abdominal signal estimates:
    #
    cwt_wdw_trans = np.transpose(cwt_wdw)
    z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
    z_cwt_xcoef_rs = z_cwt_xcoef + nusv_intercept

    # Update abdominal signal estimate:
    try:
        abdominal_est[overlap_wdw_idx : (overlap_wdw_idx + svr_wdw_lth)] = np.add(z_cwt_xcoef_rs, abdominal_est[overlap_wdw_idx: (overlap_wdw_idx + svr_wdw_lth)])
    except:
        arf = 22

    # x_idxs = np.arange(len(fetal_lead_wdw))
    # figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal", "Abdominal Estimate"))
    # figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead_wdw), row=1, col=1)
    # figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead_wdw), row=2, col=1)
    # figz.append_trace(go.Scatter(x=x_idxs, y=z_cwt_xcoef_rs), row=3, col=1)
    # figz.show()
    # time.sleep(5.0)


    # maternal_feature_vectors[n_svrs, :] = cwt_wdw.flatten()
    # maternal_fetal_feature_vectors[n_svrs, :] = np.concatenate((cwt_wdw.flatten(), cwt_wdw_fetal.flatten()), axis=None)


    # nusv_res = NuSVR(nu=0.75, C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001,
    #                  cache_size=200, verbose=False, max_iter=-1)
    # z_rbf = nusv_res.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)
    #
    # nusv_lin_coef = np.float32(nusv_res.coef_)
    # cwt_wdw_trans = np.transpose(cwt_wdw)
    # z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
    # z_cwt_xcoef_rs = np.reshape(z_cwt_xcoef, (svr_wdw_lth,)) + np.float32(nusv_res.intercept_)
    #
    # linear_regression_coefs[n_svrs, :] = np.float32(nusv_lin_coef)
    # linear_regression_intercepts[n_svrs] = np.float32(nusv_res.intercept_)


    if ((n_svrs % 50) == 1214):
        figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal",
                                                             "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4, Training Record Length = 5000",
                                                             "Abdominal Estimate"))
        # x_idxs = np.arange(len(fetal_lead))
        figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead_wdw), row=1, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead_wdw), row=2, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=z_cwt_xcoef_rs), row=3, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=z_rbf), row=3, col=1)
        figz.show()
        time.sleep(5.0)

    if ((n_svrs % 250) == 0):
        np.save('abdominal_est1k', abdominal_est, allow_pickle=False)

        x_idxs = np.arange(overlap_wdw_idx)
        figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal", "Abdominal Estimate"))
        figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead[0 : overlap_wdw_idx]), row=1, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead[0 : overlap_wdw_idx]), row=2, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=abdominal_est[0 : overlap_wdw_idx]), row=3, col=1)
        figz.show()
        time.sleep(5.0)

    if ((n_svrs % 25) == 0):
        print(['n_svrs:  ' + str(n_svrs)])

    overlap_wdw_idx = overlap_wdw_idx + 1
    n_svrs = n_svrs + 1

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



