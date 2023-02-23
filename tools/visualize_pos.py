import cv2
import numpy as np
import matplotlib
import torch
matplotlib.use('agg')
import matplotlib.pyplot as plt

batch_size = 4

gt_boxes_3d = torch.load('vis/gt_bboxes_3d.pt', map_location='cpu')
gt_pts_centers_2d = torch.load('vis/gt_pts_centers_2d.pt', map_location='cpu')
gt_labels_3d = torch.load('vis/gt_labels_3d.pt', map_location='cpu')
gt_labels_2d = torch.load('vis/gt_labels_2d.pt', map_location='cpu')
# pts_query_pos = torch.load('vis/pts_query_pos.pt', map_location='cpu')

img_query_pos_bev = torch.load('vis/img_query_pos_bev.pt', map_location='cpu').numpy()
img_vt_query_pos_bev = torch.load('vis/img_vt_query_pos_bev.pt', map_location='cpu').numpy()

# img_query_pos_bevs = []
# all_gt_box_3d_center = []
# all_gt_proj_3d_center = []
# pos_inds = []
for i in range(batch_size):
    pos_inds_i = torch.load('vis/img_pos_inds_%d.pt'%i, map_location='cpu')
    pos_vt_inds_i = torch.load('vis/pos_view_inds_%d.pt'%i, map_location='cpu')
    # pos_inds.append(pos_inds_i)
    query_pos = img_query_pos_bev[i, pos_inds_i]
    vt_query_pos = img_vt_query_pos_bev[i, pos_vt_inds_i]
    # img_query_pos_bevs.append(img_query_pos_bev[i, pos_inds_i])
    gt_boxes_3d_sample = gt_boxes_3d[i]
    gt_box_3d_center = gt_boxes_3d_sample.gravity_center.numpy()
    gt_box_3d_center = np.array(gt_box_3d_center)
    gt_box_3d_center = (gt_box_3d_center[:, :2] + 54) / 108 * 180
    # all_gt_box_3d_center.append(gt_box_3d_center)

    gt_proj_3d_center = gt_pts_centers_2d[i].numpy()
    gt_proj_3d_center = (gt_proj_3d_center[:, :2] + 54) / 108 * 180
    # all_gt_proj_3d_center.append(gt_proj_3d_center)

    # pts_pos = pts_query_pos[i]

    gt_label_3d = gt_labels_3d[i].float().numpy()
    gt_label_2d = gt_labels_2d[i].float().numpy()

    img = np.zeros([180, 180, 3], dtype=np.uint8) + 255

    gt_info_3d = np.concatenate([gt_box_3d_center, gt_label_3d.reshape(-1,1)], axis=-1)
    gt_info_2d = np.concatenate([gt_proj_3d_center, gt_label_2d], axis=-1)

    # for id in range(gt_box_3d_center.shape[0]):
    #     cv2.circle(img, (int(gt_box_3d_center[id][0]), int(gt_box_3d_center[id][1])), radius=2, color=(0,0,255))

    for id in range(gt_proj_3d_center.shape[0]):
        cv2.circle(img, (int(gt_proj_3d_center[id][0]), int(gt_proj_3d_center[id][1])), radius=1, color=(255,0,0))

    for id in range(query_pos.shape[0]):
        cv2.circle(img, (int(query_pos[id][0]), int(query_pos[id][1])), radius=1, color=(0,255,0))

    for id in range(vt_query_pos.shape[0]):
        cv2.circle(img, (int(vt_query_pos[id][0]), int(vt_query_pos[id][1])), radius=1, color=(0, 255, 255))

    # for id in range(pts_pos.shape[0]):
    #     cv2.circle(img, (int(pts_pos[id][0]), int(pts_pos[id][1])), radius=1, color=(0,255,255))

    cv2.circle(img, (90, 90), radius=3, color=(0, 0, 0))

    cv2.imwrite('vis/images/pos_%d.png'%i, img)





