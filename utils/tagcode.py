import struct

def tagcode_to_unicode(tagcode):
    return struct.pack('>H', tagcode).decode('gb2312')

def unicode_to_tagcode(tagcode_unicode):
    return struct.unpack('>H', tagcode_unicode.encode('gb2312'))[0]
