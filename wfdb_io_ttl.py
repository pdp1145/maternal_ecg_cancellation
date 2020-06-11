import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath

# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.append("/home/pdp1145/fetal_ecg_det/wfdb_python_master/")

import kymatio

import wfdb

arf = 12

# record = wfdb.rdrecord('/home/pdp1145/fetal_ecg_det/wfdb_python_master/sample-data/a103l')
record = wfdb.rdrecord('/home/pdp1145/fetal_ecg_det/fetal_ecg_data/ARR_03')
mat_lead = record.p_signal[0:1000,0]
fetal_lead = record.p_signal[0:1000,4]

x = np.arange(len(mat_lead))

fig = go.Figure(data=go.Scatter(x=x, y=fetal_lead))
fig = make_subplots(rows=2, cols=1)
fig.append_trace(go.Scatter(x=x, y=fetal_lead, row=1, col=1))
fig.append_trace(go.Scatter(x=x, y=mat_lead, row=2, col=1))
fig.show()


arf = 14

wfdb.plot_wfdb(record=record, title='Record a103l from Physionet Challenge 2015')
