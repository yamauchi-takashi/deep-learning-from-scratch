# import urllib.request
# import os
import gzip
import numpy as np

# base_url = 'http://yann.lecun.com/exdb/mnist/'
# files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']


# download_dir = 'dataset'
# os.makedirs(download_dir, exist_ok=True)

# for file in files:
#     url = base_url + file
#     save_path = os.path.join(download_dir, file)
#     urllib.request.urlretrieve(url, save_path)

# 画像データを読み込む関数
def load_images(filename):
    with gzip.open(filename, 'rb') as file:
        # マジックナンバーとヘッダー情報を読み込む
        _ = file.read(4)
        num_images = int.from_bytes(file.read(4), byteorder='big')
        num_rows = int.from_bytes(file.read(4), byteorder='big')
        num_cols = int.from_bytes(file.read(4), byteorder='big')
        
        # 画像データを読み込み
        image_data = file.read()
        
        # バイトデータをNumPyの配列に変換
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = images.reshape(num_images, num_rows*num_cols)
        
        return images

# ラベルデータを読み込む関数
def load_labels(filename):
    with gzip.open(filename, 'rb') as file:
        # マジックナンバーとヘッダー情報を読み込む
        _ = file.read(4)
        num_labels = int.from_bytes(file.read(4), byteorder='big')
        
        # ラベルデータを読み込み
        label_data = file.read()
        
        # バイトデータをNumPyの配列に変換
        labels = np.frombuffer(label_data, dtype=np.uint8)
        
        return labels

def load_mnist():
    download_dir = 'dataset'

    # ファイルパスを指定してデータを読み込む
    train_images = load_images(download_dir + '/train-images-idx3-ubyte.gz')
    train_labels = load_labels(download_dir + '/train-labels-idx1-ubyte.gz')
    test_images = load_images(download_dir + '/t10k-images-idx3-ubyte.gz')
    test_labels = load_labels(download_dir + '/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels




