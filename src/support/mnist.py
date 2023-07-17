# coding: utf-8
from pathlib import Path
from typing import Mapping, MutableMapping

from src.modules.lower_layer_modules.FileSideEffects import prepare_directory

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import gzip
import os
import os.path
import pickle

import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

prefix_directory: Path = Path(__file__).parent.parent.parent
dataset_dir: Path = prefix_directory / "data" / "MNIST"
save_file: Path = dataset_dir / "mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name: str) -> None:
    file_path: Path = dataset_dir / file_name

    if os.path.exists(file_path):
        return

    print(f"Downloading {file_name} ... ")
    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"}
    request = urllib.request.Request(f"{url_base}{file_name}", headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, mode='wb') as f:
        f.write(response)
    print("Done")


def download_mnist() -> None:
    """
    Download mnist data from url_base.
    """
    for v in key_file.values():
        _download(v)


def _load_label(file_name: str) -> np.ndarray:
    """
    Load label data from file_name.
    """
    file_path: Path = dataset_dir / file_name

    print(f"Converting {file_name} to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels: np.ndarray = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name: str) -> np.ndarray:
    """
    Load image data from file_name.
    """
    file_path: Path = dataset_dir / file_name

    print(f"Converting {file_name} to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def _convert_numpy() -> Mapping[str, np.ndarray]:
    """
    Convert mnist data to numpy array.
    """
    dataset = dict()
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def init_mnist() -> None:
    """
    Initialize mnist dataset.
    """
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(x):
    """
    Convert label data to one-hot array.
    """
    t = np.zeros((x.size, 10))
    for idx, row in enumerate(t):
        row[x[idx]] = 1

    return t


def load_mnist(
        normalize=True,
        flatten=True,
        one_hot_label=False
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    MNISTデータセットの読み込み

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
        prepare_directory(save_file.parent)
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset: MutableMapping[str, np.ndarray] = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
