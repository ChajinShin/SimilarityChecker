import torch
from src.model import VGG19
from src.dataset import get_data
from src.utils import tensor_to_numpy, Clicker, calc_coord, calc_similarity, synthesize_heatmap_and_show, get_meshgrid
from torch.nn.functional import interpolate


def Solver(opt, dev):
    # task
    if opt['task'] == 'visualization':
        # get data
        img_q, img_k = get_data(opt)

        # image size
        img_q_H, img_q_W = img_q.shape[2:]
        img_k_H, img_k_W = img_k.shape[2:]

        network = VGG19(opt['which_layer_extract']).to(dev)
        with torch.no_grad():
            query_features = network(img_q.to(dev))
            key_features = network(img_k.to(dev))

        # query image show
        plot_query_img = tensor_to_numpy(img_q)
        x_coord, y_coord = Clicker(plot_query_img).get_coord()
        x_coord, y_coord = calc_coord(img_q_H, img_q_W, query_features, x_coord, y_coord)
        h_grid, w_grid, h_size, w_size = get_meshgrid(x_coord, y_coord, 0, 0, query_features.shape[2]-1, query_features.shape[3]-1, opt['coord_size'])
        query_features = query_features[..., h_grid.long(), w_grid.long()].view(*query_features.shape[:2], h_size, w_size)

        # calc similarity
        similarity = calc_similarity(query_features, key_features).view(1, 1, *key_features.shape[2:])

        # interpolation
        similarity = interpolate(similarity, size=[img_k_H, img_k_W], mode='bilinear', align_corners=True).cpu()
        similarity = similarity.squeeze().cpu().numpy()

        # show
        # image denormalization
        img_k = (img_k + 1) / 2
        img_k = img_k.squeeze().permute(1, 2, 0).contiguous().cpu().numpy()
        synthesize_heatmap_and_show(img_k, similarity, opt['density'], opt['dataset']['result_directory'])

    elif opt['task'] == 'numeric':
        # get data
        imgs_q, imgs_k, imgs_q_name_list, imgs_k_name_list = get_data(opt)

        network = VGG19(opt['which_layer_extract']).to(dev)
        for img_q, img_k, img_q_name, img_k_name in zip(imgs_q, imgs_k, imgs_q_name_list, imgs_k_name_list):
            with torch.no_grad():
                query_features = network(img_q.to(dev))
                key_features = network(img_k.to(dev))

            # calc similarity
            similarity = calc_similarity(query_features, key_features)
            similarity = torch.mean(similarity)
            print(f"{img_q_name} - {img_k_name} :  {similarity:0.4f}")








