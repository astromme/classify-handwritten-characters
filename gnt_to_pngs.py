from libgnt.gnt import samples_from_gnt

import os
import png
import sys

def main():
    if len(sys.argv) != 3:
        print("usage: {} GNTFILE OUTPUTDIR".format(sys.argv[0]))
        print("  -> dumps images in GNTFILE to OUTPUTDIR")
        return

    _, gntfile, outputdir = sys.argv

    try:
        os.makedirs(outputdir)
    except FileExistsError:
        pass

    for i, (bitmap, character) in enumerate(samples_from_gnt(gntfile)):
        png.from_array(bitmap, 'L').save(os.path.join(outputdir, '{} {}.png'.format(character, i)))

if __name__ == "__main__":
    main()
