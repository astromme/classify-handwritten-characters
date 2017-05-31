#!/usr/bin/env python3

import os
import sys

import numpy as np

from .tagcode import tagcode_to_unicode

def samples_from_gnt(f):
    header_size = 10

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
        yield bitmap, tagcode

def read_gnt_in_directory(gnt_dirpath):
    for file_name in os.listdir(gnt_dirpath):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dirpath, file_name)
            with open(file_path, 'rb') as f:
                for bitmap, tagcode in samples_from_gnt(f):
                    yield bitmap, tagcode

def main():
    import png

    if len(sys.argv) != 3:
        print("usage: {} gntfile outputdir".format(sys.argv[0]))

    _, gntfile, outputdir = sys.argv

    try:
        os.makedirs(outputdir)
    except FileExistsError:
        pass

    with open(gntfile) as f:
        for i, (bitmap, tagcode) in enumerate(samples_from_gnt(f)):
            character = tagcode_to_unicode(tagcode)
            png.from_array(bitmap, 'L').save(os.path.join(outputdir, '{} {}.png'.format(character, i)))

if __name__ == "__main__":
    main()
