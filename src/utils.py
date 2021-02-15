import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def tensor_to_numpy(x):
    x = (x + 1) / 2
    x = x.squeeze().permute(1, 2, 0).contiguous().numpy()
    return x


def calc_coord(img_query_h, img_query_w, query_features, x_coord, y_coord):
    # coordinate에 해당하는 부분의 query 좌표를 반환
    feature_h, feature_w = query_features.shape[2:]

    # x_coord, y_coord를 [0, 1] 사이의 값으로 normalization
    x_coord = x_coord / img_query_w
    y_coord = y_coord / img_query_h

    # x_coord, y_coord를 feature 해상도에 대응시킨다.
    x_coord = round(x_coord * feature_w)
    y_coord = round(y_coord * feature_h)
    return x_coord, y_coord


def get_neighbor_coord(coord, size, min_value, max_value):
    bottom = coord - size if (coord - size) > min_value else min_value
    top = coord + size if (coord + size) < max_value else max_value
    coords = torch.arange(bottom, top+1, 1)
    return coords


def get_meshgrid(x_coord, y_coord, x_min, y_min, x_max, y_max, size):
    x_coords = get_neighbor_coord(x_coord, size, x_min, x_max)
    y_coords = get_neighbor_coord(y_coord, size, y_min, y_max)
    w_grid, h_grid = torch.meshgrid(x_coords, y_coords)
    h_size = len(y_coords)
    w_size = len(x_coords)
    return h_grid, w_grid, h_size, w_size


def calc_similarity(query_features, key_features, eps=1e-8):
    assert query_features.shape[1] == key_features.shape[1], "Channel size of two features must be same."
    b, c = query_features.shape[:2]

    query_features = query_features.view(b, c, -1)
    key_features = key_features.view(b, c, -1)

    query_features = query_features / (torch.norm(query_features, dim=1, keepdim=True) + eps)
    key_features = key_features / (torch.norm(key_features, dim=1, keepdim=True) + eps)

    similarity = torch.bmm(query_features.permute(0, 2, 1), key_features)
    similarity = torch.mean(similarity, dim=1, keepdim=True)
    return similarity


def imshow(img_array):
    plt.imshow(img_array)
    plt.show()


def synthesize_heatmap_and_show(img, heatmap, density, heatmap_save_loc=None):
    fig = plt.figure(figsize=[10, 5])
    fig.add_subplot(121)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.title("Cosine similarity value map")

    fig.add_subplot(122)
    heatmapshow = None
    heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    img = 255 * img
    img = np.uint8((img * density) + (heatmapshow * (1 - density)))

    if heatmap_save_loc is not None:
        cv2.imwrite(heatmap_save_loc, img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Heatmap image")
    plt.show()


class Clicker(object):
    def __init__(self, img_array):
        self.img_array = img_array
        self.x_coord = None
        self.y_coord = None

    def get_coord(self):
        fig = plt.figure()
        fig.add_subplot(111)
        plt.imshow(self.img_array)
        fig.canvas.mpl_connect('button_press_event', self.__event__)
        plt.show()
        return self.x_coord, self.y_coord

    def __event__(self, click):
        self.x_coord = click.xdata
        self.y_coord = click.ydata
        plt.close()


