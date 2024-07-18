r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import re
#from torch._six import container_abcs, string_classes, int_classes
import collections.abc as container_abcs
string_classes = str
int_classes = int

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        # imp-AE
        if len(batch[0]) == 2:
            w2vs = []
            w2v_means = []
            for sample in batch:
                w2v, w2v_mean = sample
                w2vs.append(sample[w2v])
                w2v_means.append(sample[w2v_mean])
            w2v_means = torch.stack(w2v_means, dim=0)
            return {"w2v":w2vs,'w2v_mean':w2v_means}
        # co-training and ablation
        elif len(batch[0]) == 3:
            images = []
            w2vs = []
            w2v_means = []
            for sample in batch:
                image, w2v, w2v_mean = sample
                images.append(sample[image])
                w2vs.append(sample[w2v])
                w2v_means.append(sample[w2v_mean])
            images = torch.stack(images, dim=0)
            w2v_means = torch.stack(w2v_means, dim=0)
            return {"image":images,"w2v":w2vs,'w2v_mean':w2v_means}
        # t-sne
        elif len(batch[0]) == 4:
            images = []
            w2vs = []
            marks = []
            for sample in batch:
                image, w2v, w2v_mean, mark = sample
                images.append(sample[image])
                w2vs.append(sample[w2v])
                marks.append(sample[mark])
            images = torch.stack(images, dim=0)
            return {'image': images, 'w2v': w2vs, 'w2v_mean': w2vs, 'mark': marks}
        #usually
        else:
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
