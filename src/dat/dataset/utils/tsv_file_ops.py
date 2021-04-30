# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import base64
import json
import os
import os.path as op

import cv2
import numpy as np
from tqdm import tqdm
from utils.miscellaneous import mkdir

from .tsv_file import TSVFile


def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except ValueError:
        return None


def load_linelist_file(linelist_file):
    if linelist_file is not None:
        line_list = []
        with open(linelist_file, 'r') as fp:
            for i in fp:
                line_list.append(int(i.strip()))
        return line_list


def tsv_writer(values, tsv_file, sep='\t'):
    mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value is not None
            # this step makes sure python2 and python3 encoded img string are the same.
            # for python2 encoded image string, it is a str class starts with "/".
            # for python3 encoded image string, it is a bytes class starts with "b'/".
            # v.decode('utf-8') converts bytes to str so the content is the same.
            # v.decode('utf-8') should only be applied to bytes class type. 
            value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
            v = '{0}\n'.format(sep.join(map(str, value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    os.rename(tsv_file_tmp, tsv_file)
    os.rename(lineidx_file_tmp, lineidx_file)


def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]


def config_save_file(tsv_file, save_file=None, append_str='.new.tsv'):
    if save_file is not None:
        return save_file
    return op.splitext(tsv_file)[0] + append_str


def get_line_list(linelist_file=None, num_rows=None):
    if linelist_file is not None:
        return load_linelist_file(linelist_file)

    if num_rows is not None:
        return [i for i in range(num_rows)]


def generate_hw_file(img_file, save_file=None):
    rows = tsv_reader(img_file)
    def gen_rows():
        for i, row in tqdm(enumerate(rows)):
            row1 = [row[0]]
            img = img_from_base64(row[-1])
            height = img.shape[0]
            width = img.shape[1]
            row1.append(json.dumps([{"height":height, "width": width}]))
            yield row1

    save_file = config_save_file(img_file, save_file, '.hw.tsv')
    tsv_writer(gen_rows(), save_file)


def generate_labelmap_file(label_file, save_file=None):
    rows = tsv_reader(label_file)
    labelmap = []
    for i, row in enumerate(rows):
        labelmap.extend(set([rect['class'] for rect in json.loads(row[1])]))
    labelmap = sorted(list(set(labelmap)))

    save_file = config_save_file(label_file, save_file, '.labelmap.tsv')
    with open(save_file, 'w') as f:
        f.write('\n'.join(labelmap))


def extract_column(tsv_file, col=1, save_file=None):
    rows = tsv_reader(tsv_file)
    def gen_rows():
        for i, row in enumerate(rows):
            row1 = [row[0], row[col]]
            yield row1

    save_file = config_save_file(tsv_file, save_file, '.col.{}.tsv'.format(col))
    tsv_writer(gen_rows(), save_file)


def remove_column(tsv_file, col=1, save_file=None):
    rows = tsv_reader(tsv_file)
    def gen_rows():
        for i, row in enumerate(rows):
            del row[col]
            yield row

    save_file = config_save_file(tsv_file, save_file, '.remove.{}.tsv'.format(col))
    tsv_writer(gen_rows(), save_file)


def generate_linelist_file(label_file, save_file=None, ignore_attrs=()):
    # generate a list of image that has labels
    # images with only ignore labels are not selected. 
    line_list = []
    rows = tsv_reader(label_file)
    for i, row in tqdm(enumerate(rows)):
        labels = json.loads(row[1])
        if labels:
            if ignore_attrs and all([any([lab[attr] for attr in ignore_attrs if attr in lab]) \
                                for lab in labels]):
                continue
            line_list.append([i])

    save_file = config_save_file(label_file, save_file, '.linelist.tsv')
    tsv_writer(line_list, save_file)


def random_drop_labels(label_file, drop_ratio, linelist_file=None, 
        save_file=None, drop_image=False):
    # randomly drop labels by the ratio
    # if drop_image is true, can drop an image by removing all labels
    # otherwise will keep at least one label for each image to make sure
    # the number of images is equal
    rows = tsv_reader(label_file)
    line_list = get_line_list(linelist_file) 
    rows_new = []
    cnt_original = 0
    cnt_new = 0
    for i, row in enumerate(rows):
        if line_list and (i not in line_list):
            row_new = [row[0], json.dumps([])]
        else:
            labels = json.loads(row[1])
            if len(labels) == 0:
                labels_new = []
            else:
                rand = np.random.random(len(labels))
                labels_new = [obj for j, obj in enumerate(labels) if rand[j]>=drop_ratio]
                if not drop_image and not labels_new:
                    # make sure there is at least one label if drop image is not allowed
                    labels_new = [labels[0]]            
            cnt_original += len(labels)
            cnt_new += len(labels_new)
            row_new = [row[0], json.dumps(labels_new)]
        rows_new.append(row_new)

    save_file = config_save_file(label_file, save_file, '.drop.{}.tsv'.format(drop_ratio))
    tsv_writer(rows_new, save_file)
    print("original labels = {}".format(cnt_original))
    print("new labels = {}".format(cnt_new))
    print("given drop_ratio = {}".format(drop_ratio))
    print("real drop_ratio = {}".format(float(cnt_original - cnt_new) / cnt_original))


def merge_two_label_files(label_file1, label_file2, save_file=None):
    rows1 = tsv_reader(label_file1)
    rows2 = tsv_reader(label_file2)

    rows_new = []
    for row1, row2 in zip(rows1, rows2):
        assert row1[0] == row2[0] 
        labels = json.loads(row1[1]) + json.loads(row2[1])
        rows_new.append([row1[0], json.dumps(labels)])
    
    save_file = config_save_file(label_file1, save_file, '.merge.tsv')
    tsv_writer(rows_new, save_file)


def is_same_keys_for_files(tsv_file1, tsv_file2, linelist_file1=None, 
        linelist_file2=None):
    # check if two files have the same keys for all rows
    tsv1 = TSVFile(tsv_file1)
    tsv2 = TSVFile(tsv_file2)
    line_list1 = get_line_list(linelist_file1, tsv1.num_rows())
    line_list2 = get_line_list(linelist_file2, tsv2.num_rows())
    assert len(line_list1) == len(line_list2)
    for idx1, idx2 in zip(line_list1, line_list2):
        row1 = tsv1.seek(idx1)
        row2 = tsv2.seek(idx2)
        if row1[0] == row2[0]:
            continue
        else:
            print("key mismatch {}-{}".format(row1[0], row2[0]))
            return False
    return True


def sort_file_based_on_keys(ref_file, tsv_file, save_file=None):
    # sort tsv_file to have the same key in each row as ref_file
    if is_same_keys_for_files(ref_file, tsv_file):
        print("file keys are the same, skip sorting")
        return tsv_file

    ref_keys = [row[0] for row in tsv_reader(ref_file)]
    all_keys = [row[0] for row in tsv_reader(tsv_file)]
    indexes = [all_keys.index(key) for key in ref_keys]
    tsv = TSVFile(tsv_file)
    def gen_rows():
        for idx in indexes:
            yield tsv.seek(idx)

    save_file = config_save_file(tsv_file, save_file, '.sorted.tsv')
    tsv_writer(gen_rows(), save_file)


def reorder_tsv_keys(in_tsv_file, ordered_keys, out_tsv_file):
    tsv = TSVFile(in_tsv_file)
    keys = [tsv.seek(i)[0] for i in tqdm(range(len(tsv)))]
    key_to_idx = {key: i for i, key in enumerate(keys)}
    def gen_rows():
        for key in tqdm(ordered_keys):
            idx = key_to_idx[key]
            yield tsv.seek(idx)
    tsv_writer(gen_rows(), out_tsv_file)


def reorder_tsv_keys_with_file(in_tsv_file, ref_tsv_file, out_tsv_file):
    ordered_keys = [row[0] for row in tsv_reader(ref_tsv_file)]
    reorder_tsv_keys(in_tsv_file, ordered_keys, out_tsv_file)


def convert_caption_json_to_tsv(caption_json_file, key_tsv_file, out_tsv_file):
    keys = [row[0] for row in tsv_reader(key_tsv_file)]
    rows_dict = {key : [] for key in keys}
    
    with open(caption_json_file, 'r') as f:
        captions = json.load(f)

    for cap in captions:
        image_id = cap['image_id']
        del cap['image_id']
        if image_id in rows_dict:
            rows_dict[image_id].append(cap)

    rows = [[key, json.dumps(rows_dict[key])] for key in keys]
    tsv_writer(rows, out_tsv_file)

 
def merge_label_fields(in_tsv1, in_tsv2, out_tsv):
    # merge the label fields for each box
    def gen_rows():
        for row1, row2 in tqdm(zip(tsv_reader(in_tsv1), tsv_reader(in_tsv2))):
            assert row1[0] == row2[0]
            label_info1 = json.loads(row1[1])
            label_info2 = json.loads(row2[1])
            assert len(label_info1) == len(label_info2)
            for lab1, lab2 in zip(label_info1, label_info2):
                lab1.update(lab2)
            yield [row1[0], json.dumps(label_info1)]
    tsv_writer(gen_rows(), out_tsv)


def remove_label_fields(in_tsv, out_tsv, remove_fields):
    if type(remove_fields) == str:
        remove_fields = [remove_fields]
    assert type(remove_fields) == list
    def gen_rows():
        for row in tqdm(tsv_reader(in_tsv)):
            label_info = json.loads(row[1])
            for lab in label_info:
                for field in remove_fields:
                    if field in lab:
                        del lab[field]
            yield [row[0], json.dumps(label_info)]
    tsv_writer(gen_rows(), out_tsv)


def random_permute_label_file(in_tsv, out_tsv):
    # take a label file as input and randomly match image
    # with the label from a different image
    tsv = TSVFile(in_tsv)
    random_index = np.random.permutation(tsv.num_rows())
    def gen_rows():
        for idx, rand_idx in enumerate(random_index):
            key = tsv.seek(idx)[0]
            labels = tsv.seek(rand_idx)[1]
            yield [key, labels]
    tsv_writer(gen_rows(), out_tsv)
    # save the random index for reference
    save_file = op.splitext(out_tsv)[0] + '.random_index.tsv'
    with open(save_file, 'w') as f:
        f.write('\n'.join([str(idx) for idx in random_index]))
