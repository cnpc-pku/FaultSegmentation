# -*- coding: utf-8 -*-
"""

# Project Name:     Fault Prediction
# File Name:        model2D
# Date:             04/26/2019 9:45 AM
# Using IDE:        PyCharm Community Edition
# Author:           Donglin Zhu
# E-mail:           zhudonglin@cnpc.com.cn
# Copyright (c) 2019, All Rights Reserved.

This is a temporary script file.
"""

import numpy as np
#import matplotlib.pyplot as plt
# import matplotlib.image as mping
import random
#import tensorflow as tf
from scipy import signal
import math
#import rickerwavelet
#from parameters import tindex, vindex, testindex

'''
Matrix=int(input("input matrix size:"))
if Matrix%2==0:
    N=int(Matrix)
else:
    N=int(Matrix+1)
Amp=int(input("input folding amplitude:"))

while Amp*2>N:
    Matrix = int(input("input larger matrix size(please larger than or equal to 2 times of folding amplitude):"))
    if Matrix % 2 == 0:
        N = int(Matrix)
    else:
        N = int(Matrix + 1)
'''
N = 200
C=100
M=20
#Amp=30
tindex=2000
vindex=500
testindex=500
def ModelGenerator(N,Indexrange,seis_path,labels_path):
    for index in range(Indexrange):
        m1 = np.zeros((N + 1, N + 1))
        m2 = np.zeros((2 * N + 1, N + 1))
        m3 = np.zeros((2 * N + 1, N + 1))
        m4 = np.zeros((2 * N + 1, 2 * N + 1))
        m5 = np.zeros((2 * N + 1, 2 * N + 1))
        m6 = np.zeros((2 * N + 1, 2 * N + 1))
        m7 = np.zeros((2 * N + 1, 2 * N + 1))
        m8 = np.zeros((2 * N + 1, 2 * N + 1))
        m9 = np.zeros((2 * N + 1, 2 * N + 1))

        a = -1 + 2 * np.random.rand(N + 1)
        b = (10 * np.random.rand(M)).astype(np.int)
        # c = (90 * np.random.rand(20)).astype(np.int)
        # b=(np.random.sample(range(1,11),10)).astype(np.int)
        c = random.sample(range(1, (N - 10)), M)

        for i in range(M):
            a[c[i]:c[i] + b[i]] = 0   #sparse ref coefficient
        x = np.linspace(0, 2 * np.pi, N + 1)
        Amprange = np.array([0, 5, 10, 15, 20, 25, 30])
        Amp = np.random.choice(Amprange, 1)
        # Amp=0
        y = Amp * np.sin(x + 2 * np.pi)  # 50
        ysin = np.rint(y)
        ysin = ysin.astype(np.int)

        # Reflection model
        for i in range(N + 1):
            for j in range(N + 1):
                m1[i][j] = a[i]

        # folding
        for i in range(N + 1):
            for j in range(N + 1):
                m2[i + int(N / 2)][j] = m1[i][j]  # 50 need revise

        for i in range(int(N / 2), int(3 * N / 2 + 1), 1):  # 50,151
            for j in range(N + 1):
                m3[i + ysin[j]][j] = m2[i][j]

        for i in range(2 * N + 1):
            for j in range(N + 1):
                m4[i][j + int(N / 2)] = m3[i][j]  # 50

        # fault

        krange = [i for i in range(-89, 91)]   #generate slop(dip of fault)
        krange = np.array(krange)

        def f(x, N):
            # k = np.random.choice(krange, 1)
            fx = -1 * math.tan(np.pi / 2 - k * np.pi / 180) * x + N * (
                    1 + math.tan(np.pi / 2 - k * np.pi / 180))  # angle [-89,-45]&[45,90]

            ff = np.rint(fx)
            ff = ff.astype(np.int)
            return ff

        kc = np.random.choice(krange, 1)
        # file=open('E:/PyCharm/Practice3/dip.txt','a')
        # file.write(str(k))
        # kc=-25
        # print('kc='+str(kc))
        # print('Amp='+str(Amp))
        if kc in range(0, 45):
            k = kc - 90
            m22 = np.rot90(m4, 1)
        elif kc in range(-45, 0):
            k = kc + 90
            m22 = np.rot90(m4, 1)
        else:
            k = kc
            m22 = m4

        for i in range(2 * N + 1):
            for j in range(f(i, N)):
                m5[i][j] = m22[i][j]
        for i in range(2 * N + 1):
            for j in range(f(i, N), 2 * N + 1, 1):
                m6[i][j] = m22[i][j]

        # displacement=int(input("input fault displacement:"))
        displacement = 15
        dy = displacement / math.tan(abs(k) * np.pi / 180)
        ddy = np.rint(dy)
        ddy = ddy.astype(np.int)
        if k >= 0:
            for i in range(2 * N + 1 - displacement):
                for j in range(2 * N, ddy - 1, -1):
                    m7[i][j] = m5[i + displacement][j - ddy]
            for i in range(2 * N + 1):
                for j in range(2 * N + 1):
                    m9[i][j] = m7[i][j] + m6[i][j]

        else:
            for i in range(2 * N + 1 - displacement):
                for j in range(2 * N + 1 - ddy):
                    m8[i][j] = m5[i + displacement][j + ddy]
            for i in range(2 * N + 1):
                for j in range(2 * N + 1):
                    m9[i][j] = m8[i][j] + m6[i][j]
        # crop
        m10 = np.zeros((2 * C + 1, 2 * C + 1))
        for i in range(2 * C + 1):
            for j in range(2 * C + 1):
                m10[i][j] = m9[i + N - C][j + N - C]  #
        # plt.figure('crop_seis')  # , figsize=(6.4,9.6),dpi=10
        # plt.imshow(m10, cmap=plt.cm.gray)

        # fault_aug
        m26 = m10
        drange = np.array([[0, 5], [2, 4], [3, 0]])   #generate 1,3 or 5 trace fault
        d = random.choice(drange)

        if k >= 0:
            for i in range(4, 2 * C + 1 - 4):
                for j in range(4, 2 * C + 1 - 4):
                    # m26[i-3][f(i,C) - 3] = 6 * m9[i-3][f(i,C) - 3] / 8
                    m26[i - 2][f(i, C) - 2] = d[1] * m9[i - 2][f(i, C) - 2] / 5
                    m26[i - 1][f(i, C) - 1] = d[0] * m9[i][f(i, C) - 1] / 5
                    m26[i][f(i, C)] = 0
                    m26[i + 1][f(i, C) + 1] = d[0] * m9[i + 1][f(i, C) + 1] / 5
                    m26[i + 2][f(i, C) + 2] = d[1] * m9[i + 2][f(i, C) + 2] / 5
                    # m26[i+3][f(i,C) + 3] = 6 * m9[i+3][f(i,C) + 3] / 8
        else:
            for i in range(4, 2 * C + 1 - 4):
                for j in range(4, 2 * C + 1 - 4):
                    # print(f(i))
                    # m26[i][f(i)-7]=4*m9[i][f(i)-7]/5
                    # m26[i-6][f(i) + 6] = 4 * m9[i-6][f(i) +6] / 5
                    # m26[i-5][f(i) + 5] = 4 * m9[i-5][f(i) + 5] / 5
                    # m26[i-4][f(i)+4]=4*m9[i-4][f(i)+4]/5
                    # m26[i-3][f(i,C) + 3] = 6 * m9[i-3][f(i,C) + 3] / 8
                    m26[i - 2][f(i, C) + 2] = d[1] * m9[i - 2][f(i, C) + 2] / 5
                    m26[i - 1][f(i, C) + 1] = d[0] * m9[i][f(i, C) + 1] / 5
                    m26[i][f(i, C)] = 0
                    m26[i + 1][f(i, C) - 1] = d[0] * m9[i + 1][f(i, C) - 1] / 5
                    m26[i + 2][f(i, C) - 2] = d[1] * m9[i + 2][f(i, C) - 2] / 5
                    # m26[i+3][f(i,C) - 3] = 6 * m9[i+3][f(i,C) - 3] / 8
                    # m26[i+4][f(i) - 4] = 4*m9[i+4][f(i) - 4] / 5
                    # m26[i+5][f(i) -5] = 4 * m9[i+5][f(i) - 5] / 5
                    # m26[i+6][f(i) - 6] = 4*m9[i+6][f(i) - 6] / 5
                    # m26[i][f(i) + 7] = 4 * m9[i][f(i) + 7] / 5
        if kc in range(-45, 45):
            m23 = np.rot90(m26, -1)
        else:
            m23 = m26
        # convolve

        #from scipy import signal
        rrange = np.linspace(0.5, 3, 6)    #random choose wavelet
        rc = np.random.choice(rrange)
        r = signal.ricker(100, rc)
        '''
        import numpy as np
        import matplotlib.pyplot as plt


        def ricker(f, length=0.128, dt=0.001):
            t = np.arange(-length / 2, (length - dt) / 2, dt)
            y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t ** 2))
            return t, y
        f = 25  # A low wavelength of 25 Hz
        t, w = ricker(f)
        '''
        # plt.figure('wavelet')
        # plt.plot(r)
        m11 = np.zeros((2 * C + 1, 2 * C + 1))
        m20 = np.rot90(m23, 1)
        for i in range(2 * C + 1):
            seis = signal.convolve(m20[i], r, mode='same')  # rickerwavelet.r
            m11[i] = seis
        m21 = np.rot90(m11, -1)

        # Noise
        def wgn(x, snr):
            snr = 10 ** (snr / 10.0)
            xpower = np.sum(x ** 2) / len(x)
            npower = xpower / snr
            return np.random.randn(len(x)) * np.sqrt(npower)

        m12 = np.zeros((2 * C + 1, 2 * C + 1))
        for i in range(2 * C + 1):
            noise = wgn(m11[i], 10)
            m12[i] = m11[i] + noise
        mn = np.rot90(m12, -1)
        m12 = mn

        # crop
        m25 = np.zeros((C, C))
        for i in range(C):
            for j in range(C):
                m25[i][j] = m12[i + C - int(C / 2)][j + C - int(C / 2)]
        np.save(seis_path + str(index) + '.npy', m25)
        # label
        m13 = np.zeros((2 * C + 1, 2 * C + 1))
        for i in range(1, 2 * C):
            for j in range(1, 2 * C):
                m13[i][f(i, C) - 1] = 1
                m13[i][f(i, C)] = 1
                m13[i][f(i, C) + 1] = 1
        if kc in range(-45, 45):
            m24 = np.rot90(m13, -1)
        else:
            m24 = m13
        # crop
        m14 = np.zeros((C, C))
        for i in range(C):
            for j in range(C):
                m14[i][j] = m24[i + C - int(C / 2)][j + C - int(C / 2)]
        np.save(labels_path + str(index) + '.npy', m14)

trains=ModelGenerator(N,tindex,'images/train_gen/seis/','images/train_gen/labels/')
validation=ModelGenerator(N,vindex,'images/validation/seis/','images/validation/labels/')
test=ModelGenerator(N,testindex,'images/test/seis/','images/test/labels/')
