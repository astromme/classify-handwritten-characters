#!/usr/bin/env python3

import os
import numpy as np

from .tagcode import tagcode_to_unicode

def samples_from_gnt(gnt_filepath):
    """
    Given a gnt file path,
    returns generater that yields samples of (bitmap, character)
    """
    header_size = 10

    with open(gnt_filepath, 'rb') as f:
        # read samples from f until no bytes remaining
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break

            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            assert header_size + width*height == sample_size

            bitmap = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            yield bitmap, tagcode_to_unicode(tagcode)

def samples_from_directory(dirpath):
    """
    Given a directory path,
    Returns generator that yields samples of (bitmap, character)
    From all .gnt files in that directory.
    """

    for file_name in os.listdir(dirpath):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(dirpath, file_name)
            for bitmap, character in samples_from_gnt(file_path):
                yield bitmap, character
