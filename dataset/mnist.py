# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


base_url = 'http://yann.lecun.com/exdb/mnist/'
key_file = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

img_dim = (1, 28, 28)  ## (色, 縦, 横)
num_classes = 10  # テストラベルの数（クラスの数）を定義

def download_mnist():
    for file in key_file:
        url = base_url + file
        save_path = os.path.join(dataset_dir, file)
        urllib.request.urlretrieve(url, save_path)

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    with gzip.open(file_path, 'rb') as file:
        # マジックナンバーとヘッダー情報を読み込む
        _ = file.read(4)
        _ = int.from_bytes(file.read(4), byteorder='big') # ラベル数
        
        # ラベルデータを読み込み
        label_data = file.read()
    
    # バイトデータをNumPyの配列に変換
    print("Converting " + file_name + " to NumPy Array ...")
    labels = np.frombuffer(label_data, dtype=np.uint8)
    print("Done")

    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    with gzip.open(file_path, 'rb') as file:
        # マジックナンバーとヘッダー情報を読み込む
        _ = file.read(4)
        _ = int.from_bytes(file.read(4), byteorder='big')  # イメージ数
        num_rows = int.from_bytes(file.read(4), byteorder='big')
        num_cols = int.from_bytes(file.read(4), byteorder='big')
        img_size = num_rows * num_cols
        # 画像データを読み込み
        image_data = file.read()
        
    # バイトデータをNumPyの配列に変換
    print("Converting " + file_name + " to NumPy Array ...")
    images = np.frombuffer(image_data, dtype=np.uint8)
    images = images.reshape(-1, img_size)
    print("Done")

    return images

def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file[0])
    dataset['train_label'] = _load_label(key_file[1])
    dataset['test_img'] = _load_img(key_file[2])
    dataset['test_label'] = _load_label(key_file[3])

    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(labels):
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels] = 1
    return one_hot


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNISTデータセットの読み込み

    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
         for key in ('train_img', 'test_img'):
            # dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
            dataset[key] = dataset[key].reshape(-1, img_dim)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
