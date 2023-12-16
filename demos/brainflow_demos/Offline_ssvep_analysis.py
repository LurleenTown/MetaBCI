# -*- coding: utf-8 -*-
"""
SSVEP offline analysis.
"""
from scipy.signal import sosfiltfilt, cheby1, cheb1ord, hilbert
from metabci.brainda.algorithms.decomposition import FBTDCA, FBTRCA
from sklearn.base import BaseEstimator, ClassifierMixin
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut, EnhancedStratifiedKFold)
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.feature_analysis.time_freq_analysis \
    import TimeFrequencyAnalysis
from metabci.brainda.algorithms.feature_analysis.freq_analysis \
    import FrequencyAnalysis
from metabci.brainda.algorithms.feature_analysis.time_analysis \
    import TimeAnalysis
from datasets import MetaBCIData
from mne.filter import resample
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy
import numpy as np
import warnings
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')


# 对raw操作,例如滤波

def raw_hook(raw, caches):
    # do something with raw object
    # raw.filter(1, None, phase='zero-double')
    # raw.filter(7, 55, l_trans_bandwidth=2, h_trans_bandwidth=5,
    #            phase='zero-double')
    # caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches


# 按照0,1,2,...重新排列标签


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


class MaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y


def get_ITR(N=0, P=1.0, T=1.66):
    if P == 1:
        B = np.log2(N) + P * np.log2(P)
    else:
        B = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))

    ITR = B * 60 / T
    return ITR


# 绘制FFT图


def draw_fft(y):
    blue_black = [26 / 255, 49 / 255, 139 / 255]
    blue_white = [130 / 255, 170 / 255, 231 / 255]
    red_black = [192 / 255, 63 / 255, 103 / 255]
    red_white = [247 / 255, 166 / 255, 191 / 255]
    # fft #
    N = 1000
    fft_y = np.abs(fft(y))
    normalization__half_c = fft_y[range(int(N / 10))]
    x_ = np.arange(N)
    half_x = x_[range(int(N / 10))]

    plt.subplot(211)
    plt.plot(x_, y, color=blue_black)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.title('Band-passed Filtered Waveform')
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude[μV]")

    plt.subplot(212)
    # plt.stem(half_x, normalization__half_c, 'blue')
    fren_max = np.argmax(normalization__half_c)
    plt.plot(half_x, normalization__half_c, color=blue_white)
    plt.plot(fren_max, normalization__half_c[fren_max], "o", color=red_black)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.title('Spectrogram')
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Amplitude[μV]")
    plt.show()


# 获取滤波后数据


def get_filter_data(X):
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    wp = [
        [6, 88], [14, 88], [22, 88], [30, 88], [38, 88], [46, 88]
    ]
    ws = [
        [4, 90], [12, 90], [20, 90], [28, 90], [36, 90], [44, 90]
    ]
    # wp = [
    #     [4, 90], [12, 90], [20, 90], [28, 90], [36, 90], [44, 90]
    # ]
    # ws = [
    #     [2, 92], [10, 92], [18, 92], [26, 92], [34, 92], [42, 92]
    # ]
    # wp = [
    #     [4, 45], [12, 45], [20, 45], [28, 45], [36, 45], [44, 45]
    # ]
    # ws = [
    #     [2, 47], [10, 47], [18, 47], [26, 47], [34, 47], [42, 47]
    # ]

    filterweights = np.arange(1, 7) ** (-2.0) + 1.5
    filterbank = generate_filterbank(wp, ws, 1000)
    Xs = np.stack([sosfiltfilt(sos, X, axis=-1) for sos in filterbank])
    return Xs


# 训练模型


def train_model(X, y, srate=1000):
    # print("train the model")
    y = np.reshape(y, (-1))
    # 降采样
    # X = resample(X, up=256, down=srate)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    # 滤波器组设置
    wp = [
        [6, 88], [14, 88], [22, 88], [30, 88], [38, 88], [46, 88]
    ]
    ws = [
        [4, 90], [12, 90], [20, 90], [28, 90], [36, 90], [44, 90]
    ]

    # filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterweights = np.arange(1, 7) ** (-2.0) + 1.5
    filterbank = generate_filterbank(wp, ws, 1000)

    freqs = np.arange(8, 16, 0.4)
    Yf = generate_cca_references(freqs, srate=1000, T=0.5, n_harmonics=5)
    model = FBTRCA(filterbank, n_components=1, ensemble=True,
                   filterweights=np.array(filterweights))
    model = model.fit(X, y)

    return model


# 预测标签


def model_predict(X, srate=1000, model=None):
    # print("predict")
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    # X = resample(X, up=256, down=srate)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # FBDSP.predict()预测标签
    p_labels = model.predict(X)
    return p_labels


# 计算离线正确率


def offline_validation(X, y, srate=1000):
    # print("Start validation")
    y = np.reshape(y, (-1))

    kfold_accs = []
    # spliter = EnhancedLeaveOneGroupOut(return_validate=False)       # 留一法交叉验证
    spliter = EnhancedStratifiedKFold(return_validate=False)  # K折交叉验证

    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(X_train, y_train, srate=srate)  # 训练模型
        p_labels = model_predict(X_test, srate=srate, model=model)  # 预测标签
        kfold_accs.append(np.mean(p_labels == y_test))  # 记录正确率
    return np.mean(kfold_accs)


# 时域分析


def time_feature(X, meta, dataset, event, channel, latency=0):
    # brainda.algorithms.feature_analysis.time_analysis.TimeAnalysis
    Feature_R = TimeAnalysis(X, meta, dataset, event=event, latency=latency,
                             channel=channel)  # trial*channel*sample

    plt.figure(1, figsize=(6, 4))
    # 计算模板信号调用TimeAnalysis.stacking_average()
    data_mean = Feature_R.stacking_average(np.squeeze(
        Feature_R.data[:, Feature_R.chan_ID, :]), _axis=0)

    draw_fft(data_mean)

    # 绘制plv图
    # t = np.linspace(0, 1, int(1 * 1000))
    # data_ref = np.sin(2 * np.pi * 9 * t + np.pi * 0)
    # mean_plv = get_plv(data_ref, 8, 0)
    # mean_plv = get_plv(data_mean, 8, 0)
    # print(mean_plv)
    # ax = plt.subplot(2, 1, 1)
    sample_num = int(Feature_R.fs * Feature_R.data_length)
    # 画出模板信号及其振幅调用TimeAnalysis.plot_single_trial()
    # loc, amp, ax = Feature_R.plot_single_trial(data_mean,
    #                                            sample_num=sample_num,
    #                                            axes=ax,
    #                                            amp_mark='peak',
    #                                            time_start=0,
    #                                            time_end=sample_num - 1)
    # plt.title("(a)", x=0.03, y=0.86)
    # 画出多试次信号调用TimeAnalysis.plot_multi_trials()
    ax = plt.subplot(1, 1, 1)
    ax = Feature_R.plot_multi_trials(
        np.squeeze(Feature_R.data[:, Feature_R.chan_ID, :]),
        sample_num=sample_num, axes=ax)
    # plt.title("(b)", x=0.03, y=0.86)

    # 时域幅值脑地形图
    # fig2 = plt.figure(2)
    # data_map = Feature_R.stacking_average(Feature_R.data, _axis=0)
    # # 调用TimeAnalysis.plot_topomap()
    # Feature_R.plot_topomap(data_map, loc, fig=fig2,
    #                        channels=Feature_R.All_channel, axes=ax)
    plt.show()


# 频域分析


def frequency_feature(X, chan_names, event, SNRchannels, plot_ch, srate=1000):
    # 初始化参数
    # channellist = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    channellist = ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
                   'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    chan_nums = []
    for i in range(len(chan_names)):
        chan_nums.append(channellist.index(chan_names[i]))
    X = X[:, chan_nums, :]
    SNRchannels = chan_names.index(SNRchannels)

    # brainda.algorithms.feature_analysis.freq_analysis.FrequencyAnalysis
    Feature_R = FrequencyAnalysis(X, meta, event, srate)

    # 计算模板信号,调用FrequencyAnalysis.stacking_average()
    mean_data = Feature_R.stacking_average(data=[], _axis=0)

    # 计算12Hz刺激下模板信号的功率谱密度
    # 调用FrequencyAnalysis.power_spectrum_periodogram()
    f, den = Feature_R.power_spectrum_periodogram(mean_data[plot_ch])
    plt.plot(f, den)
    plt.text(12, den[f == 12][0], '{:.2f}'.format(
        den[f == 12][0]), fontsize=15)
    plt.text(24, den[f == 24][0], '{:.2f}'.format(
        den[f == 24][0]), fontsize=15)
    plt.text(36, den[f == 36][0], '{:.2f}'.format(
        den[f == 36][0]), fontsize=15)
    plt.title('OZ FFT')
    plt.xlim([0, 60])
    plt.ylim([0, 4])
    plt.xlabel('fre [Hz]')
    plt.ylabel('PSD [V**2]')
    plt.show()


def time_frequency_feature(X, y, chan_names, srate=1000):
    # 初始化参数
    channellist = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
                   'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
                   'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                   'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
                   'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                   'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
                   'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2']
    chan_nums = []
    for i in range(len(chan_names)):
        chan_nums.append(channellist.index(chan_names[i]))
    X = X[:, chan_nums, :]
    index_8hz = np.where(y == 0)
    data_8hz = np.squeeze(X[index_8hz, :, :])
    mean_data_8hz = np.mean(data_8hz, axis=0)
    fs = srate

    # brainda.algorithms.feature_analysis.time_freq_analysis.TimeFrequencyAnalysis
    Feature_R = TimeFrequencyAnalysis(fs)

    # 短时傅里叶变换
    nfft = mean_data_8hz.shape[1]
    # 调用TimeFrequencyAnalysis.fun_stft()
    f, t, Zxx = Feature_R.fun_stft(
        mean_data_8hz, nperseg=256, axis=1, nfft=nfft)
    Zxx_Pz = Zxx[-4, :, :]
    plt.pcolormesh(t, f, np.abs(Zxx_Pz))
    plt.ylim(0, 25)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.show()

    # 莫雷小波变换
    mean_Pz_data_8hz = mean_data_8hz[-4, :]
    N = mean_Pz_data_8hz.shape[0]
    t_index = np.linspace(0, N / fs, num=N, endpoint=False)
    omega = 2
    sigma = 1
    data_test = np.reshape(mean_Pz_data_8hz, newshape=(
        1, mean_Pz_data_8hz.shape[0]))
    # 调用TimeFrequencyAnalysis.func_morlet_wavelet()
    P, S = Feature_R.func_morlet_wavelet(data_test, f, omega, sigma)
    f_lim = np.array([min(f[np.where(f > 0)]), 30])
    f_idx = np.array(np.where((f <= f_lim[1]) & (f >= f_lim[0])))[0]
    t_lim = np.array([0, 1])
    t_idx = np.array(
        np.where((t_index <= t_lim[1]) & (t_index >= t_lim[0])))[0]
    PP = P[0, f_idx, :]
    plt.pcolor(t_index[t_idx], f[f_idx], PP[:, t_idx])
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency(Hz)')
    plt.xlim(t_lim)
    plt.ylim(f_lim)
    plt.plot([0, 0], [0, fs / 2], 'w--')
    plt.title(
        ''.join(
            ('Scaleogram (ω = ', str(omega), ' , ', 'σ = ', str(sigma), ')')
        ))
    plt.text(t_lim[1] + 0.04, f_lim[1] / 2,
             'Power (\muV^2/Hz)', rotation=90,
             verticalalignment='center',
             horizontalalignment='center')
    plt.colorbar()
    plt.show()

    # 希尔伯特变换
    charray = np.mean(data_8hz, axis=1)
    tarray = charray[0, :]
    N1 = tarray.shape[0]
    # 调用TimeFrequencyAnalysis.fun_hilbert()
    analytic_signal, realEnv, imagEnv, angle, envModu = Feature_R.fun_hilbert(
        tarray)

    time = np.linspace(0, N1 / fs, num=N1, endpoint=False)
    plt.plot(time, realEnv, "k", marker='o',
             markerfacecolor='white', label=u"real part")
    plt.plot(time, imagEnv, "b", label=u"image part")
    plt.plot(time, angle, "c", linestyle='-', label=u"angle part")
    plt.plot(time, analytic_signal, "grey", label=u"signal")
    plt.ylabel('Angle or amplitude')
    plt.legend()
    plt.show()


def get_plv(f_x, t_freqs, t_phases, T=1, srate=1000):
    t = np.linspace(0, T, int(T * srate))
    f_template = np.sin(2 * np.pi * t_freqs * t + np.pi * t_phases)

    amp_phi = np.angle(hilbert(f_x))  # 脑电信号振幅的相位
    phase = np.angle(hilbert(f_template))
    plv = np.abs(np.sum(np.exp(1j * (phase - amp_phi))) / len(phase))

    # plt.plot(len(phase), observed_plv, 'blue')
    # plt.title('PLV', fontsize=9, color='blue')
    # plt.show()
    return plv


if __name__ == '__main__':
    # 初始化参数
    # 放大器的采样率
    srate = 1000
    n_target = 15
    cue_length = 1
    sub_name = 'jhz'
    pad_name = 'ssvep-vr-'
    data_length = 0.25
    for i in range(1, 5):
        # 截取数据的时间段
        stim_interval = [(0.14, 0.14 + data_length * i)]
        # subjects = list(range(1, 2))

        subjects = list((sub_name + '\\' + pad_name,))
        paradigm = 'ssvep'
        # pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
        pick_chs = ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'CB2',
                    'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2', 'CB1']
        # //.datasets.py中按照metabci.brainda.datasets数据结构自定义数据类MetaBCIData
        # declare the dataset
        dataset = MetaBCIData(  # 读取离线数据
            subjects=subjects, srate=srate,
            paradigm='ssvep', pattern='ssvep')
        paradigm = SSVEP(
            channels=dataset.channels,
            events=dataset.events,
            intervals=stim_interval,
            srate=srate)
        paradigm.register_raw_hook(raw_hook)
        # 调用paradigms\base.py中BaseParadigm类的get_data()函数 --> 该函数调用self._get_single_subject_data()函数 --> 该函数调用brainda\datasets\base.py中BaseDataset类的get_data()函数
        # --> 该函数调用brainflow_demos\datasets.py中子类MetaBCIData的_get_single_subject_data()函数 --> 该函数调用了该类的self.data_path函数获取cnt文件中的数据
        X, y, meta = paradigm.get_data(
            dataset,
            subjects=subjects,
            return_concat=True,
            n_jobs=4,
            verbose=False)
        y = label_encoder(y, np.unique(y))
        print("Loding data successfully")
        # scipy.io.savemat('./data/vr/hp-data.mat', {'hp_data': X})
        # X_filter = get_filter_data(X)  # sub-band*trial*channel*sample
        # scipy.io.savemat('./data/vr/bp-data.mat', {'bp_data': X_filter})
        # 计算离线正确率
        acc = offline_validation(X, y, srate=srate)  # 计算离线准确率
        print("Current Model accuracy:{:.2f}".format(acc))
        itr = get_ITR(n_target, acc, cue_length + data_length * i)
        print("Current Model ITR:{:.2f}".format(itr))
        if i == 1:
            answer_data = DataFrame({'姓名': [sub_name], '范式': [pad_name], '数据长度': [data_length * i], 'ACC': [acc], 'ITR': [itr]})
        else:
            answer_data.loc[i] = [sub_name, pad_name, data_length * i, acc, itr]
    org_data = pd.read_excel('AnalysisAnswer.xlsx', sheet_name='sheet1')
    if org_data.size == 0:
        answer_data.to_excel('AnalysisAnswer.xlsx', sheet_name='sheet1', index=False)
    else:
        save_data = pd.concat([org_data, answer_data], axis=0)
        # save_data =  org_data.append(answer_data, ignore_index=True)
        save_data.to_excel('AnalysisAnswer.xlsx', sheet_name='sheet1', index=False)

    # 时域分析
    # time_feature(X[..., :int(srate)], meta, dataset, '8', ['PO4'])  # 1s
    # time_feature(X_filter[0, ..., :int(srate)], meta, dataset, '8', ['OZ'])  # 1s
    # 频域分析
    # frequency_feature(X[..., :int(srate)], pick_chs, '1', 'OZ', -2, srate)
    # 时频域分析
    # time_frequency_feature(X[...,:srate], y,pick_chs)
