# -*- coding:utf-8 -*-
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# x = np.linspace(0, 1, 1005)
c_list = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0]*67
c_array = np.array(c_list)
fft_c = fft(c_array)
N = 1005
x = np.arange(N)
half_x = x[range(int(N / 2))]

abs_c = np.abs(fft_c)
angle_c = np.angle(fft_c)
normalization_c = abs_c / N
normalization__half_c = normalization_c[range(int(N / 2))]

plt.subplot(221)
plt.plot(x, c_array)
plt.title('原始波形')

plt.subplot(222)
plt.plot(x, fft_c, 'black')
plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')

plt.subplot(223)
plt.plot(x, abs_c, 'r')
plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')

# plt.subplot(234)
# plt.plot(x, angle_c, 'violet')
# plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')

# plt.subplot(223)
# plt.plot(x, normalization_c, 'g')
# plt.title('双边振幅谱(归一化)', fontsize=9, color='green')

plt.subplot(224)
plt.plot(half_x, normalization__half_c, 'blue')
plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')

plt.show()
