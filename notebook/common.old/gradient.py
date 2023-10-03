import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np

def numerical_gradient(f, x):
    h = 1e-4   # 0.0001 微小な変化量（刻み幅）を設定
    grad = np.zeros_like(x)    # xと同じ形状のゼロベクトルを作成し、勾配を初期化
    
    # xを多次元配列として反復処理するためのイテレータを作成
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index   # インデックスを取得
        tmp_val = x[idx]       # xの現在の値を一時的に保存
        x[idx] = tmp_val + h   # xの要素をhだけ増やして、
        fxh1 = f(x)            # 関数fを評価（f(x+h)の計算）
        
        x[idx] = tmp_val - h   # xの要素をhだけ減らして、
        fxh2 = f(x)            # 関数fを評価（f(x-h)の計算）
        
        grad[idx] = (fxh1 - fxh2) / (2*h)  # 勾配の計算（数値微分の定義に従って）
        
        x[idx] = tmp_val       # xの要素を元に戻す
        it.iternext()          # イテレータを次に進める
        
    # 計算された勾配ベクトルを返す
    return grad
