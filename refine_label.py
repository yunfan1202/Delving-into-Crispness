import os
import cv2
import numpy as np
import torch
from networks import EdgeGenerator
from tqdm import tqdm
import torchvision.transforms.functional as F
from PIL import Image
from skimage import measure


def set_connectivity_value(edge_edge, min_thresh):
    labels = measure.label(edge_edge, connectivity=2)
    props = measure.regionprops(labels)
    for each_area in props:
        if each_area.area <= min_thresh:
            for each_point in each_area.coords:
                edge_edge[each_point[0]][each_point[1]] = 0.2 * 255  # set a value < beta
    return edge_edge


def get_canny_fuse(img_data, t_min=20, t_max=40):
    def use_more_canny(imgs, t_min, t_max):
        for index, each_img in enumerate(imgs):
            if index == 0:
                canny_final = cv2.Canny(each_img, t_min, t_max)
            else:
                canny_tmp = cv2.Canny(each_img, t_min, t_max)
                canny_final = cv2.add(canny_final, canny_tmp)
        return canny_final
    img_blur_g = cv2.GaussianBlur(img_data, (5, 5), 0)
    img_blur_g_b1 = cv2.bilateralFilter(img_blur_g, 15, 50, 50)
    img_blur_g_b2 = cv2.bilateralFilter(img_blur_g_b1, 15, 50, 50)
    imgs_data = [img_blur_g, img_blur_g_b1, img_blur_g_b2]
    canny_ranges = [(t_min, t_max)]
    for j, canny_range in enumerate(canny_ranges):
        canny_fuse = use_more_canny(imgs_data, canny_range[0], canny_range[1])
    return canny_fuse


def get_patches(mask_data, patch_size, top_k=5):
    labels = measure.label(mask_data, connectivity=2)
    props = measure.regionprops(labels)
    # print("total area:", len(props))
    areas = []
    bboxes = []
    coords_all = []
    for each_area in props:
        areas.append(each_area.area)
        bboxes.append(each_area.bbox)
        coords_all.append(each_area.coords)
    Z = zip(areas, bboxes, coords_all)
    Z = sorted(Z, reverse=True)
    # areas_sort, bboxes_sort,  coords_all_sort = zip(*Z)
    # print(areas_sort)

    index = 0
    patches = []
    # new_masks = []
    for area, bbox, coords in Z:
        if index == top_k:
            break
        index += 1
        x = bbox[1]
        y = bbox[0]
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)

        height, width = mask_data.shape[:2]

        if width > center_x + int(patch_size / 2):
            left_top_x = max(0, center_x - int(patch_size / 2))
        else:
            left_top_x = max(0, width - patch_size)

        if height - center_y + int(patch_size / 2) > 0:
            left_top_y = max(0, center_y - int(patch_size / 2))
        else:
            left_top_y = max(0, height - patch_size)

        right_down_x = min(width, left_top_x + patch_size)
        right_down_y = min(height, left_top_y + patch_size)

        right_down_x = left_top_x + (right_down_x - left_top_x) // 8 * 8
        right_down_y = left_top_y + (right_down_y - left_top_y) // 8 * 8
        patches.append([left_top_x, right_down_x, left_top_y, right_down_y])
    return len(props), patches


def get_mask(edge_data, lb_data):
    def is_Mask(edge, h, w, size=3):
        assert size % 2 != 0
        r = size // 2
        Neighbors = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                Neighbors.append((i, j))
        for neighbor in Neighbors:
            dr, dc = neighbor
            try:
                if edge[h + dr][w + dc] != 0:
                    return False
            except IndexError:
                pass
        return True

    if len(edge_data.shape) != 2:
        edge_data = cv2.cvtColor(edge_data, cv2.COLOR_BGR2GRAY)
    if len(lb_data.shape) != 2:
        lb_data = cv2.cvtColor(lb_data, cv2.COLOR_BGR2GRAY)
    lb_data[lb_data < 0.3 * 255] = 0
    lb_data[lb_data >= 0.3 * 255] = 255
    height, width = lb_data.shape[:2]

    mask_data = np.zeros((height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            if lb_data[h, w] > 0:
                if is_Mask(edge_data, h, w, size=5):
                    mask_data[h, w] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_data = cv2.dilate(mask_data, kernel, iterations=2)
    return mask_data


def canny_overlap(lb_data, canny_fuse, beta=0.3):
    def filter_canny_connectivity(canny_edge, min_thresh):
        labels = measure.label(canny_edge, connectivity=2)
        props = measure.regionprops(labels)
        # print("total area:", len(props))

        for each_area in props:
            if each_area.area <= min_thresh:
                for each_point in each_area.coords:
                    canny_edge[each_point[0]][each_point[1]] = 0
        return canny_edge

    if len(lb_data.shape) != 2:
        lb_data = cv2.cvtColor(lb_data, cv2.COLOR_BGR2GRAY)
    lb_data[lb_data < beta * 255] = 0
    lb_data[lb_data >= beta * 255] = 255
    lb_data = cv2.dilate(lb_data, kernel=np.ones((3, 3), np.uint8), iterations=1)

    # canny_fuse = get_canny_fuse(img_data)
    overlap = cv2.bitwise_and(canny_fuse, lb_data)
    overlap = filter_canny_connectivity(overlap, min_thresh=5)
    return overlap


def last_refine(edge_data, lb_data):
    def is_neighbor_big(edge, h, w, size=5):
        assert size % 2 != 0
        r = size // 2
        Neighbors = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                Neighbors.append((i, j))
        for neighbor in Neighbors:
            dr, dc = neighbor
            try:
                if edge[h + dr][w + dc] >= 0.3 * 255:
                    return True
            except IndexError:
                pass
        return False

    def is_neighbor_empty(edge, h, w, size=3):
        assert size % 2 != 0
        r = size // 2
        Neighbors = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                Neighbors.append((i, j))
        for neighbor in Neighbors:
            dr, dc = neighbor
            try:
                if edge[h + dr][w + dc] != 0:
                    return False
            except IndexError:
                pass
        return True

    if len(edge_data.shape) != 2:
        edge_data = cv2.cvtColor(edge_data, cv2.COLOR_BGR2GRAY)
    if len(lb_data.shape) != 2:
        lb_data = cv2.cvtColor(lb_data, cv2.COLOR_BGR2GRAY)

    height, width = lb_data.shape[:2]

    temp_data = edge_data.copy()
    temp_data[temp_data < 0.1 * 255] = 0   # set some pixels that are very small but !=0 to 0
    for h in range(height):
        for w in range(width):
            if lb_data[h, w] > 0:
                if is_neighbor_empty(temp_data, h, w, size=3):
                    edge_data[h, w] = 0.2 * 255  # just set a value < beta
                if (temp_data[h, w] < 0.3 * 255) and is_neighbor_big(temp_data, h, w, size=5):
                    edge_data[h, w] = 0

    return edge_data


def get_imgs_list(imgs_dir):
    imgs_list = [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]
    imgs_list.sort()
    return imgs_list


def test(model, img_gray, edge_data, mask_data, device):
    def postprocess(img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def to_tensor(img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    # load image and record the raw height and width
    h_raw, w_raw = img_gray.shape[:2]
    h = h_raw // 8 * 8
    w = w_raw // 8 * 8

    img_gray = cv2.resize(img_gray, (w, h))
    img_gray = to_tensor(img_gray).unsqueeze(0).to(device)

    # process edge
    edge_data = cv2.resize(edge_data, (w, h), interpolation=cv2.INTER_NEAREST)
    edge_data = to_tensor(edge_data).unsqueeze(0).to(device)
    # process mask
    mask_data = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_data = (mask_data > 0).astype(np.uint8) * 255
    mask_data = to_tensor(mask_data).unsqueeze(0).to(device)

    # ----------------generate edges--------------
    edge_masked = (edge_data * (1 - mask_data))
    image_masked = (img_gray * (1 - mask_data)) + mask_data
    inputs = torch.cat((image_masked, edge_masked, mask_data), dim=1)
    outputs = model(inputs)
    # print(img_gray.shape, mask.shape, edge.shape)

    outputs_merged = (outputs * mask_data) + (edge_data * (1 - mask_data))
    output = postprocess(outputs_merged)[0]
    output = output.cpu().numpy().astype(np.uint8).squeeze()
    output = cv2.resize(output, (w_raw, h_raw))
    return output


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        device = torch.device("cpu")
        print("No GPU!")

    # build the model and initialize
    model = EdgeGenerator(use_spectral_norm=True).to(device)
    gen_weights_path = "./EdgeModel_gen.pth"
    data = torch.load(gen_weights_path)
    model.load_state_dict(data['generator'])

    images_path = "datasets/BSDS_example/image"
    label_path = "datasets/BSDS_example/edge_raw"
    results_path = "datasets/BSDS_example/edge_refine"

    os.makedirs(results_path, exist_ok=True)

    print('\nstart testing...\n')

    imgs = get_imgs_list(images_path)
    for index in tqdm(range(len(imgs))):
        name = os.path.basename(imgs[index])[:-4] + ".png"
        print(name)
        img_data = cv2.imread(imgs[index])
        img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        lb_data = cv2.imread(os.path.join(label_path, name), 0)
        temp_lb_data = lb_data.copy()

        canny_fuse = get_canny_fuse(img_data, t_min=20, t_max=40)
        edge_data = canny_overlap(lb_data, canny_fuse)

        # # --------------patch-based iterative edge refinement---------------
        max_iter = 30
        min_iter = 5
        current_area_num = 1e6  # just set a big number
        for iter_num in range(max_iter):
            mask_data = get_mask(edge_data, lb_data)

            area_num, patches = get_patches(mask_data, patch_size=256, top_k=1000)
            if current_area_num <= area_num and iter_num >= min_iter:
            # if current_area_num <= area_num:
                print("stop in iteration " + str(iter_num) + " with area number " + str(current_area_num))
                break
            current_area_num = area_num
            print("current area number:", current_area_num)
            for each_patch in patches:
                [p_x, p_x_end, p_y, p_y_end] = each_patch
                img_gray_patch = img_gray[p_y:p_y_end, p_x:p_x_end]
                edge_data_patch = edge_data[p_y:p_y_end, p_x:p_x_end]
                mask_data_patch = mask_data[p_y:p_y_end, p_x:p_x_end]
                output_edge = test(model, img_gray_patch, edge_data_patch, mask_data_patch, device)
                temp_patch = edge_data[p_y:p_y_end, p_x:p_x_end]
                temp_patch = cv2.add(temp_patch, output_edge)
                edge_data[p_y:p_y_end, p_x:p_x_end] = temp_patch
                edge_data = cv2.bitwise_and(canny_fuse, edge_data)

        edge_data = set_connectivity_value(edge_data, min_thresh=5)
        last_edge = last_refine(edge_data, temp_lb_data)
        cv2.imwrite(os.path.join(results_path, name), last_edge)

if __name__ == "__main__":
    main()
