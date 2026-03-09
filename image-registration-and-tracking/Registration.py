import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import os
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


RESULTS_DIR = './results'


def save_figure(filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()


def find_match(img1, img2):
    # To do
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        x1 = np.zeros((0, 2))
        x2 = np.zeros((0, 2))
        return x1, x2

    ratio = 0.75

    nbrs12 = NearestNeighbors(n_neighbors=min(2, des2.shape[0]), algorithm='auto')
    nbrs12.fit(des2)
    dist12, idx12 = nbrs12.kneighbors(des1)
    if dist12.shape[1] == 1:
        keep12 = np.ones(des1.shape[0], dtype=bool)
    else:
        keep12 = dist12[:, 0] < ratio * dist12[:, 1]
    match12 = -np.ones(des1.shape[0], dtype=int)
    match12[keep12] = idx12[keep12, 0]

    nbrs21 = NearestNeighbors(n_neighbors=min(2, des1.shape[0]), algorithm='auto')
    nbrs21.fit(des1)
    dist21, idx21 = nbrs21.kneighbors(des2)
    if dist21.shape[1] == 1:
        keep21 = np.ones(des2.shape[0], dtype=bool)
    else:
        keep21 = dist21[:, 0] < ratio * dist21[:, 1]
    match21 = -np.ones(des2.shape[0], dtype=int)
    match21[keep21] = idx21[keep21, 0]

    matched_idx1 = []
    matched_idx2 = []
    for i, j in enumerate(match12):
        if j >= 0 and match21[j] == i:
            matched_idx1.append(i)
            matched_idx2.append(j)

    x1 = np.array([kp1[i].pt for i in matched_idx1], dtype=np.float64)
    x2 = np.array([kp2[i].pt for i in matched_idx2], dtype=np.float64)
    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    def estimate_affine(src, dst):
        n = src.shape[0]
        M = np.zeros((2 * n, 6), dtype=np.float64)
        b = np.zeros((2 * n,), dtype=np.float64)
        M[0::2, 0] = src[:, 0]
        M[0::2, 1] = src[:, 1]
        M[0::2, 2] = 1
        M[1::2, 3] = src[:, 0]
        M[1::2, 4] = src[:, 1]
        M[1::2, 5] = 1
        b[0::2] = dst[:, 0]
        b[1::2] = dst[:, 1]
        p, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
        A_est = np.array([[p[0], p[1], p[2]],
                          [p[3], p[4], p[5]],
                          [0, 0, 1]], dtype=np.float64)
        return A_est

    if x1.shape[0] < 3:
        A = np.eye(3, dtype=np.float64)
        return A

    n = x1.shape[0]
    best_inliers = None
    best_count = -1
    best_error = np.inf
    rng = np.random.default_rng(0)

    x1_h = np.hstack((x1, np.ones((n, 1), dtype=np.float64)))

    for _ in tqdm(range(int(ransac_iter)), desc='RANSAC', leave=False):
        sample_idx = rng.choice(n, size=3, replace=False)
        x1_sample = x1[sample_idx]
        x2_sample = x2[sample_idx]
        if np.linalg.matrix_rank(np.hstack((x1_sample, np.ones((3, 1))))) < 3:
            continue
        if np.linalg.matrix_rank(np.hstack((x2_sample, np.ones((3, 1))))) < 3:
            continue
        A_try = estimate_affine(x1_sample, x2_sample)

        x2_pred = x1_h @ A_try[:2, :].T
        err = np.linalg.norm(x2_pred - x2, axis=1)
        inliers = err < ransac_thr
        inlier_count = np.sum(inliers)

        if inlier_count > best_count:
            best_inliers = inliers
            best_count = inlier_count
            if inlier_count > 0:
                best_error = np.mean(err[inliers])
            else:
                best_error = np.inf
        elif inlier_count == best_count and inlier_count > 0:
            this_error = np.mean(err[inliers])
            if this_error < best_error:
                best_inliers = inliers
                best_error = this_error

    if best_inliers is None or np.sum(best_inliers) < 3:
        A = estimate_affine(x1, x2)
    else:
        A = estimate_affine(x1[best_inliers], x2[best_inliers])
        x2_refit = x1_h @ A[:2, :].T
        refit_err = np.linalg.norm(x2_refit - x2, axis=1)
        refit_inliers = refit_err < ransac_thr
        if np.sum(refit_inliers) >= 3:
            A = estimate_affine(x1[refit_inliers], x2[refit_inliers])
    return A

def warp_image(img, A, output_size):
    # To do
    h_out, w_out = output_size[0], output_size[1]
    ys, xs = np.meshgrid(np.arange(h_out, dtype=np.float64),
                         np.arange(w_out, dtype=np.float64),
                         indexing='ij')
    coords = np.stack((xs.ravel(), ys.ravel(), np.ones(xs.size, dtype=np.float64)), axis=1)
    mapped = coords @ A.T
    mapped_x = mapped[:, 0] / mapped[:, 2]
    mapped_y = mapped[:, 1] / mapped[:, 2]

    query_points = np.stack((mapped_y, mapped_x), axis=1)
    img_warped = interpolate.interpn(
        (np.arange(img.shape[0], dtype=np.float64), np.arange(img.shape[1], dtype=np.float64)),
        img.astype(np.float64),
        query_points,
        method='linear',
        bounds_error=False,
        fill_value=0
    ).reshape(h_out, w_out)
    return img_warped


def align_image(template, target, A):
    # To do
    template_f = template.astype(np.float64) / 255.0
    target_f = target.astype(np.float64) / 255.0

    h, w = template_f.shape
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64),
                         indexing='ij')

    grad_y, grad_x = np.gradient(template_f)

    sd = np.stack((
        grad_x * xs,
        grad_x * ys,
        grad_x,
        grad_y * xs,
        grad_y * ys,
        grad_y
    ), axis=2).reshape(-1, 6)

    H = sd.T @ sd
    H_inv = np.linalg.pinv(H)
    coords = np.stack((xs.ravel(), ys.ravel(), np.ones(h * w, dtype=np.float64)), axis=1)

    A_refined = A.astype(np.float64).copy()
    errors = []
    max_iter = 200
    eps = 1e-4

    for _ in tqdm(range(max_iter), desc='IC Align', leave=False):
        warped = warp_image(target_f, A_refined, template_f.shape)
        mapped = coords @ A_refined.T
        valid = (
            (mapped[:, 0] >= 0) & (mapped[:, 0] <= target_f.shape[1] - 1) &
            (mapped[:, 1] >= 0) & (mapped[:, 1] <= target_f.shape[0] - 1)
        )
        if not np.any(valid):
            break

        err_flat = (warped - template_f).reshape(-1)
        valid_err = np.zeros_like(err_flat)
        valid_err[valid] = err_flat[valid]
        errors.append(np.mean(np.abs(err_flat[valid])))

        F = sd.T @ valid_err
        delta_p = H_inv @ F

        delta_A = np.array([
            [1 + delta_p[0], delta_p[1], delta_p[2]],
            [delta_p[3], 1 + delta_p[4], delta_p[5]],
            [0, 0, 1]
        ], dtype=np.float64)

        A_refined = A_refined @ np.linalg.inv(delta_A)

        if np.linalg.norm(delta_p) < eps:
            break

    A_refined = (A_refined, np.array(errors))
    return A_refined


def track_multi_frames(template, img_list):
    # To do
    A_list = []
    if len(img_list) == 0:
        return A_list

    x1, x2 = find_match(template, img_list[0])
    A_init = align_image_using_feature(x1, x2, 5, 3000)
    A_first = align_image(template, img_list[0], A_init)
    if isinstance(A_first, tuple):
        A_first = A_first[0]
    A_list.append(A_first)

    template_curr = warp_image(img_list[0], A_first, template.shape)
    A_prev = A_first

    for i in tqdm(range(1, len(img_list)), desc='Track Frames', leave=False):
        A_now = align_image(template_curr, img_list[i], A_prev)
        if isinstance(A_now, tuple):
            A_now = A_now[0]
        A_list.append(A_now)
        template_curr = warp_image(img_list[i], A_now, template.shape)
        A_prev = A_now
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    plt.figure(figsize=(12, 6))
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    save_figure('find_match.png')

def visualize_align_image(template, target, A, A_refined, errors=None):
    plt.figure(figsize=(14, 8))
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    save_figure('align_image.png')

    if errors is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        save_figure('align_error_plot.png')


def visualize_track_multi_frames(template, img_list, A_list):
    plt.figure(figsize=(10, 8))
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    save_figure('track_multi_frames.png')


if __name__ == '__main__':
    template = cv2.imread('./JS_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in tqdm(range(4), desc='Load Frames', leave=False):
        target = cv2.imread('./JS_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 5
    ransac_iter = 3000
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    plt.figure(figsize=(6, 6))
    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    save_figure('warp_image.png')

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)
