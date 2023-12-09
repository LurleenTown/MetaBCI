from metabci.brainflow.amplifiers import LSLInlet, DataInlet, MarkerInlet
from typing import List
import pylsl
import numpy as np
import math


class LSLapps():
    """An amplifier implementation for Lab streaming layer (LSL) apps.
    LSL ref as: https://github.com/sccn/labstreaminglayer
    The LSL provides many builded apps for communiacting with varities
    of devices, and some of the are EEG acquiring device, like EGI, g.tec,
    DSI and so on. For metabci, here we just provide a pathway for reading 
    lsl data streams, which means as long as the the LSL providing the app, 
    the metabci could support its online application. Considering the 
    differences among different devices for tranfering the event trigger. 
    YOU MUST BE VERY CAREFUL to determine wether the data stream reading 
    from the LSL apps contains a event channel. For example, the neuroscan
    synamp II will append a extra event channel to the raw data channel.
    Because we do not have chance to test each device that LSL supported, so 
    plese modifiy this class before using with your own condition.
    """

    def __init__(self, ):
        streams = pylsl.resolve_streams()
        self.marker_inlet = None
        self.data_inlet = None
        self.device_data = None
        self.marker_data = None
        self.marker_cache = np.zeros((5, 2))
        self.marker_count = 0
        for info in streams:
            if info.type() == 'Markers':
                if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                        or info.channel_format() != pylsl.cf_string:
                    print('Invalid marker stream ' + info.name())
                print('Adding marker inlet: ' + info.name())
                self.marker_inlet = MarkerInlet(info)
            elif info.nominal_srate() != pylsl.IRREGULAR_RATE \
                    and info.channel_format() != pylsl.cf_string:
                print('Adding data inlet: ' + info.name())
                self.data_inlet = DataInlet(info)
            else:
                print('Don\'t know what to do with stream ' + info.name())

    def save_marker(self):
        n_marker = self.marker_data.shape[0]
        self.marker_cache[self.marker_count:self.marker_count + n_marker]
        self.marker_count += n_marker

    def recv(self):
        self.device_data = self.data_inlet.stream_action()
        self.marker_data = self.marker_inlet.stream_action()
        if any(self.marker_data):
            self.save_marker()
        if any(self.device_data) and any(self.marker_cache):
            self.save_marker()
            label_column = np.zeros((self.device_data.shape[0], 1))
            insert_keys = self.device_data[:, -1].searchsorted(
                self.marker_cache[:, -1])
            label_column[insert_keys] = self.marker_cache[:, 1]
            self.device_data[:, -1] = label_column
            self.marker_cache = np.zeros((5, 2))
            self.marker_count = 0
            return self.device_data.tolist()
        elif any(self.device_data):
            self.device_data[:, -1] = 0
            return self.device_data.tolist()
        elif any(self.marker_data):
            self.save_marker()
            return []
        else:
            return []


def get_ITR(N=0, P=1, T=1.66):
    if P == 1:
        B = math.log2(N) + P * math.log2(P)
    else:
        B = math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))
    ITR = B * 60 / T
    return ITR


code_title = ['cvep-cs', 'ssvep-cs', 'cvep-vr', 'ssvep-vr']
# sub_acc = [[0.92, 0.99, 0.92, 0.97], [0.95, 0.98, 0.92, 0.98], [0.82, 0.98, 0.94, 0.97], [0.90, 0.94, 0.91, 0.92], [0.80, 0.92, 0.87, 0.98]]
sub_acc = [[[0.66, 0.83, 0.88, 0.92], [0.80, 0.98, 0.99, 0.99], [0.76, 0.86, 0.91, 0.92], [0.71, 0.91, 0.95, 0.97]],
           [[0.72, 0.88, 0.94, 0.95], [0.75, 0.94, 0.97, 0.98], [0.74, 0.87, 0.91, 0.92], [0.78, 0.93, 0.98, 0.98]],
           [[0.56, 0.72, 0.79, 0.82], [0.84, 0.97, 0.98, 0.98], [0.76, 0.88, 0.93, 0.94], [0.81, 0.94, 0.96, 0.97]],
           [[0.58, 0.73, 0.84, 0.90], [0.52, 0.80, 0.88, 0.94], [0.73, 0.79, 0.87, 0.91], [0.62, 0.82, 0.91, 0.92]],
           [[0.45, 0.61, 0.73, 0.80], [0.53, 0.79, 0.87, 0.92], [0.52, 0.72, 0.83, 0.87], [0.60, 0.89, 0.95, 0.98]]]
itr = np.zeros((5, 4, 4))
for sub_index in range(0, 5):
    for code_index in range(0, 4):
        for data_index in range(0, 4):
            itr[sub_index, code_index, data_index] = get_ITR(N=15, P=sub_acc[sub_index][code_index][data_index], T=1 + 0.25 * (data_index + 1))

print(itr)
