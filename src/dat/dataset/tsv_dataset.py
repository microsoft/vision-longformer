# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import os
import base64
from io import BytesIO
import json
from PIL import Image
import torch.utils.data as data

from .utils.tsv_file import TSVFile
from .utils.load_files import load_linelist_file, load_from_yaml_file
from .utils.load_files import find_file_path_in_yaml, load_labelmap_file


class TSVDataset(data.Dataset):
    def __init__(self, img_file, label_file=None, hw_file=None, linelist_file=None,
                 labelmap_file=None, transforms=None):
        """
        Dataset Constructor from TSV Files
            Input:
                img_file: Image file with image key and base64 encoded image str.
            label_file: An optional label file with image key and label information.
                A label_file is required for training and optional for testing.
            hw_file: An optional file with image key and image height/width info.
            linelist_file: An optional file with a list of line indexes to load samples.
                It is useful to select a subset of samples or duplicate samples.
        """
        self.img_file = img_file
        self.label_file = label_file
        self.hw_file = hw_file
        self.linelist_file = linelist_file
        self.labelmap_file = labelmap_file
        self.transforms = transforms

        self.img_tsv = TSVFile(img_file)
        self.label_tsv = None if label_file is None else TSVFile(label_file)
        self.hw_tsv = None if hw_file is None else TSVFile(hw_file)
        self.line_list = load_linelist_file(linelist_file)
        self.labelmap = load_labelmap_file(labelmap_file)

    def __len__(self):
        if self.line_list is None:
            return self.img_tsv.num_rows()
        else:
            return len(self.line_list)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        annotations = self.get_annotations(idx)
        target = self.get_target_from_annotations(annotations)
        img, target = self.apply_transforms(img, target)
        return img, target, idx

    def get_image(self, idx):
        line_no = self.get_line_no(idx)
        row = self.img_tsv.seek(line_no)
        # use -1 to support old format with multiple columns.
        img = Image.open(BytesIO(base64.b64decode(row[-1])))
        return img.convert('RGB')

    def get_line_no(self, idx):
        return idx if self.line_list is None else self.line_list[idx]

    def get_annotations(self, idx):
        line_no = self.get_line_no(idx)
        if self.label_tsv is not None:
            row = self.label_tsv.seek(line_no)
        else:
            row = self.img_tsv.seek(line_no)
        anno = row[1]
        if anno.startswith('{') or anno.startswith('['):
            annotations = json.loads(anno)
        else:
            annotations = anno
        return annotations

    def get_target_from_annotations(self, annotations):
        # This function will be overwritten by each dataset to
        # decode the labels to specific formats for each task.
        if self.labelmap is not None and type(annotations)==str:
            annotations = self.labelmap[annotations]
        return int(annotations)

    def apply_transforms(self, image, target=None):
        # This function will be overwritten by each dataset to
        # apply transforms to image and targets.
        if self.transforms is not None:
            image = self.transforms(image)
        return image, target

    def get_img_key(self, idx):
        line_no = self.get_line_no(idx)
        # based on the overhead of reading each row.
        if self.hw_tsv:
            return self.hw_tsv.seek(line_no)[0]
        elif self.label_tsv:
            return self.label_tsv.seek(line_no)[0]
        else:
            return self.img_tsv.seek(line_no)[0]


class TSVYamlDataset(TSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """
    def __init__(self, yaml_file, transforms=None):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = os.path.dirname(yaml_file)
        img_file = find_file_path_in_yaml(self.cfg['img'], self.root)
        label_file = find_file_path_in_yaml(self.cfg.get('label', None),
                                            self.root)
        hw_file = find_file_path_in_yaml(self.cfg.get('hw', None), self.root)
        linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
                                               self.root)
        labelmap_file = find_file_path_in_yaml(self.cfg.get("labelmap", None),
                                               self.root)

        super(TSVYamlDataset, self).__init__(
            img_file, label_file, hw_file, linelist_file, labelmap_file,
            transforms=transforms
        )