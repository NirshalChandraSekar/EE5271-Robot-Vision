import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

def get_differential_filter():
    # To do
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    im_h, im_w = im.shape
    filter_h, filter_w = filter.shape
    pad_h, pad_w = filter_h // 2, filter_w // 2
    im_padded = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    im_filtered = np.zeros_like(im)
    for i in range(im_h):
        for j in range(im_w):
            im_filtered[i, j] = np.sum(im_padded[i:i+filter_h, j:j+filter_w] * filter)

    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do
    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    grad_angle = np.arctan2(im_dy, im_dx) % np.pi

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    num_bins = 6
    im_h, im_w = grad_mag.shape
    num_cell_h, num_cell_w = im_h // cell_size, im_w // cell_size
    ori_histo = np.zeros((num_cell_h, num_cell_w, num_bins))
    bin_width = np.pi / num_bins
    bin_offset = bin_width / 2.0
    for i in range(num_cell_h):
        for j in range(num_cell_w):
            cell_mag = grad_mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_angle = grad_angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            bin_idx = np.floor(((cell_angle + bin_offset) % np.pi) / bin_width).astype(int)
            histo = np.bincount(bin_idx.ravel(), weights=cell_mag.ravel(), minlength=num_bins)
            ori_histo[i, j, :] = histo

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    num_cell_h, num_cell_w, num_bins = ori_histo.shape
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    ori_histo_normalized = np.zeros((num_blocks_h, num_blocks_w, num_bins * block_size**2))
    eps = 0.001
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block_histo = ori_histo[i:i+block_size, j:j+block_size, :].reshape(-1)
            norm = np.sqrt(np.sum(block_histo**2) + eps)
            ori_histo_normalized[i, j, :] = block_histo / norm

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do

    # get differential images using get_differential_filter and filter_image
    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)

    # Compute gradients using get_gradient
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)

    # Build histogram of oriented gradients for all cells using build_histogram
    cell_size = 8
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)

    # Build the descriptor of all blocks with normalization using get_block_descriptor
    block_size = 2
    hog = get_block_descriptor(ori_histo, block_size)

    # return a long vector by concatenating all the block descriptors
    hog = hog.flatten()

    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    if im.max() > 1:
        im = im.astype('float') / 255.0
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    # plt.show()
    plt.savefig('hog_visualization.png', dpi=300)


def face_recognition(I_target, I_template):
    # To do
    cell_size = 8
    block_size = 2
    ncc_threshold = 0.35
    iou_threshold = 0.5

    # convert to float in [0, 1]
    I_target = I_target.astype('float') / 255.0
    I_template = I_template.astype('float') / 255.0

    # compute gradients once
    filter_x, filter_y = get_differential_filter()
    target_dx = filter_image(I_target, filter_x)
    target_dy = filter_image(I_target, filter_y)
    target_mag, target_angle = get_gradient(target_dx, target_dy)

    template_dx = filter_image(I_template, filter_x)
    template_dy = filter_image(I_template, filter_y)
    template_mag, template_angle = get_gradient(template_dx, template_dy)

    def hog_from_grad(grad_mag, grad_angle):
        ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
        hog = get_block_descriptor(ori_histo, block_size)
        return hog.reshape(-1)

    # template descriptor (zero-mean)
    template_hog = hog_from_grad(template_mag, template_angle)
    template_hog = template_hog - np.mean(template_hog)
    template_norm = np.linalg.norm(template_hog)
    if template_norm < 1e-6:
        return np.zeros((0, 3))

    temp_h, temp_w = I_template.shape
    targ_h, targ_w = I_target.shape

    boxes = []
    for y in range(targ_h - temp_h + 1):
        for x in range(targ_w - temp_w + 1):
            patch_mag = target_mag[y:y+temp_h, x:x+temp_w]
            patch_angle = target_angle[y:y+temp_h, x:x+temp_w]
            patch_hog = hog_from_grad(patch_mag, patch_angle)
            patch_hog = patch_hog - np.mean(patch_hog)
            denom = np.linalg.norm(patch_hog) * template_norm + 1e-6
            score = np.dot(patch_hog, template_hog) / denom
            if score >= ncc_threshold:
                boxes.append([x, y, score])

    if len(boxes) == 0:
        return np.zeros((0, 3))

    boxes = np.array(boxes, dtype=np.float32)

    # Non-maximum suppression with IoU threshold
    scores = boxes[:, 2]
    order = scores.argsort()[::-1]
    keep = []
    box_size = temp_h
    box_area = float(box_size * box_size)
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 0] + box_size, boxes[rest, 0] + box_size)
        yy2 = np.minimum(boxes[i, 1] + box_size, boxes[rest, 1] + box_size)
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = box_area + box_area - inter
        iou = inter / (union + 1e-6)
        order = rest[iou < iou_threshold]

    bounding_boxes = boxes[keep]
    return bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    # plt.figure(3)
    # plt.imshow(fimg, vmin=0, vmax=1)
    # plt.show()
    # save the visualization result
    cv2.imwrite('face_detection_result.png', fimg)




if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)
    visualize_hog(im, hog, 8, 2)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.
