import os
import json

from options import get_options
from datasets import get_dataloader
from model import get_model
import utils
import h5py
import numpy as np


COMP_CAT_DICT_PATH = 'tools/clevr_comp_cat_dict.json'


opt = get_options('test')
test_loader = get_dataloader(opt, 'test')
model = get_model(opt)

if opt.use_cat_label:
    with open(COMP_CAT_DICT_PATH) as f:
        cat_dict = utils.invert_dict(json.load(f))
if opt.split == 'val' or 'test':
    num_img = 15000
else:
    num_img = 70000

if opt.dataset == 'clevr':
    scenes = [{
        'image_index': i,
        'image_filename': 'CLEVR_%s_%06d.png' %(opt.split, i),
        'objects': []
    } for i in range(num_img)]

count = 0

for data, _, idxs, cat_idxs in test_loader:
    model.set_input(data)
    model.forward()
    pred = model.get_pred()
    for i in range(pred.shape[0]):
        img_id = idxs[i]
        object = utils.softmax_attribute(pred[i])
        if img_id >= num_img:
            input(img_id)
        scenes[img_id]['objects'].append(object)
    count += idxs.size(0)
    print('%d / %d objects processed' % (count, len(test_loader.dataset)))

##0:3 shape; 3:5 size; 5:7 materials; 7:15 color; 15:18 position

features = []
for i in range(num_img):
    feature = np.asarray(scenes[i]['objects'])
    filled = np.zeros((15, 18))
    filled[: feature.shape[0]] = feature
    features.append(filled)

print('| saving annotation file to %s' % opt.output_path)
utils.mkdirs(os.path.dirname(opt.output_path))
with h5py.File(opt.output_path, 'w') as f:
    f.create_dataset('features', data=np.asarray(features), dtype='float32')

print('| finish')
