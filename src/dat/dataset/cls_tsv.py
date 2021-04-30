# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import base64
from io import BytesIO
import json
from PIL import Image
from .tsv_dataset import TSVYamlDataset


class ClsTsvDataset(TSVYamlDataset):
    """
    Generic TSV dataset format for Classification.
    """
    def __init__(self, yaml_file, transforms=None, **kwargs):
        super(ClsTsvDataset, self).__init__(yaml_file, transforms=transforms)
        assert self.label_tsv is None

    def __getitem__(self, idx):
        line_no = self.get_line_no(idx)
        row = self.img_tsv.seek(line_no)
        # get image
        # use -1 to support old format with multiple columns.
        img = Image.open(BytesIO(base64.b64decode(row[-1])))
        img = img.convert('RGB')
        # get target
        annotations = json.loads(row[1])
        target = annotations[0]['class']
        if self.labelmap is not None:
            target = self.labelmap[target]
        img, target = self.apply_transforms(img, int(target))
        return img, target, idx

