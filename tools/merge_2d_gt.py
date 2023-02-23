import os
import pickle
import json
import numpy as np

cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

data_root = "data/nuscenes/"

info_file = "nuscenes_infos_val.pkl"
coco_file = "nuscenes_infos_val.coco.json"

output_file = "nuscenes_infos_val_with_gt2d.pkl"

info_path = os.path.join(data_root, info_file)
coco_file_path = os.path.join(data_root, coco_file)

with open(info_path, "rb") as file:
    info = pickle.load(file, encoding="bytes")

with open(coco_file_path, "r") as file:
    coco = json.load(file)

print("Generate id to ann file")
id2ann = {}
for ann in coco["annotations"]:
    image_id = ann['image_id']
    if image_id not in id2ann:
        id2ann[image_id] = [ann]
    else:
        id2ann[image_id].append(ann)

print("%d images with ann"%len(id2ann))

for image in coco["images"]:
    image_id = image['id']
    if image_id not in id2ann:
        id2ann[image_id] = []
print("total %d images"%len(id2ann))


print("Generate new info file")
for info_id, info_dict in enumerate(info['infos']):
    if info_id % 200 == 1:
        print(info_id, "/", len(info['infos']))
    bboxes = []
    gt_names = []
    view_ids = []
    for view_id, cam in enumerate(cam_orders):
        cam_dict = info_dict['cams'][cam]
        image_token = cam_dict['sample_data_token']
        camera_anns = id2ann[image_token]

        for ann in camera_anns:
            view_ids.append(view_id)
            bboxes.append(ann['bbox'])
            gt_names.append(ann['category_name'])

    view_ids = np.array(view_ids)
    bboxes = np.array(bboxes).reshape(-1, 4)

    info_dict['gt_bboxes_2d'] = bboxes
    info_dict['gt_names_2d'] = gt_names
    info_dict['gt_views_2d'] = view_ids

    import pdb
    pdb.set_trace()

# output_path = os.path.join(data_root, output_file)
# with open(output_path, "wb") as file:
#     pickle.dump(info, file)
