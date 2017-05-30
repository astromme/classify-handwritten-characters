#!/usr/bin/env python3

import os
import struct
import sys

import numpy as np

def read_gnt_in_directory(gnt_dirpath):
    def samples(f):
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

    for file_name in os.listdir(gnt_dirpath):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dirpath, file_name)
            with open(file_path, 'rb') as f:
                for bitmap, tagcode in samples(f):
                    yield bitmap, tagcode

def read_pot_in_directory(pot_dirpath):
    def samples(f):
        header_size = 8

        # read samples from f until no bytes remaining
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break

            sample_size = header[0] + (header[1]<<8)
            tagcode = header[2] + (header[3]<<8) # + (header[3] << 16) + (header[2] << 24)
            stroke_number = header[6] + (header[7]<<8)
            #assert header_size + width*height == sample_size

            strokes = []
            stroke = []

            xmin = sys.maxsize
            ymin = sys.maxsize

            xmax = -sys.maxsize
            ymax = -sys.maxsize

            while True:
                x, y = np.fromfile(f, dtype='int16', count=2)
                if x == -1 and y == -1:
                    break
                elif x == -1:
                    strokes.append(stroke)
                    stroke = []
                else:
                    stroke.append((x, y))
                    xmin = min(xmin, x)
                    ymin = min(ymin, y)

                    xmax = max(xmax, x)
                    ymax = max(ymax, y)

            height = ymax - ymin
            width = xmax - xmin

            longer_side = max(height, width)

            #print('h: {}, w: {}, x: {}-{}, y: {}-{}'.format(height, width, xmin, xmax, ymin, ymax))

            def normalize_stroke(xmin, ymin, longer_side, stroke):
                def normalize_point(xmin, ymin, longer_side, point):
                    if len(point) < 2:
                        return point
                    oldx, oldy = point

                    x = (oldx - xmin) / longer_side - 0.5
                    y = (oldy - ymin) / longer_side - 0.5

                    return (x, y)

                return [normalize_point(xmin, ymin, longer_side, point) for point in stroke]
            strokes = [normalize_stroke(xmin, ymin, longer_side, stroke) for stroke in strokes]

            yield strokes, tagcode

    for file_name in os.listdir(pot_dirpath):
        if file_name.endswith('.pot'):
            file_path = os.path.join(pot_dirpath, file_name)
            with open(file_path, 'rb') as f:
                for strokes, tagcode in samples(f):
                    yield strokes, tagcode


def tagcode_to_unicode(tagcode):
    return struct.pack('>H', tagcode).decode('gb2312')

def unicode_to_tagcode(tagcode_unicode):
    return struct.unpack('>H', tagcode_unicode.encode('gb2312'))[0]


def main():
    import svgwrite

    i = 0
    for strokes, tagcode in read_pot_in_directory('OLHWDB1.1tst_pot/'):
        character = tagcode_to_unicode(tagcode)
        print(character, len(strokes))
        dwg = svgwrite.Drawing('{} {}.svg'.format(i, character), profile='tiny')
        for stroke in strokes:
            for point1, point2 in zip(stroke, stroke[1:]):
                point1 = [int(point1[0]/10), int(point1[1]/10)]
                point2 = [int(point2[0]/10), int(point2[1]/10)]
                dwg.add(dwg.line(point1, point2, stroke=svgwrite.rgb(10, 10, 16, '%')))
        dwg.save(pretty=True)
        i += 1

if __name__ == "__main__":
    main()
