import os
from PIL import Image

yourpath = os.getcwd() + '/output_patch_temp/'
outpath = os.getcwd() + '/output_patch_jpg/'

def convert_to_jpeg(img_name):
    yourpath = os.getcwd() + '/output_patch_temp/'
    outpath = os.getcwd() + '/output_patch_jpg/' + img_name

    for root, dirs, files in os.walk(yourpath, topdown=False):
        for name in files:
            #print(os.path.join(outpath, name))
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
                if os.path.isfile(os.path.splitext(os.path.join(outpath, name))[0] + ".jpg"):
                    print(f"A jpeg file already exists for {name}")
                # If a jpeg is *NOT* present, create one from the tiff.
                else:
                    outfile = os.path.splitext(os.path.join(outpath, name))[0] + ".jpg"
                    try:
                        im = Image.open(os.path.join(root, name))
                        print("Generating jpeg for %s" % name)
                        im.thumbnail(im.size)
                        im.save(outfile, "JPEG", quality=100)
                    except(Exception, e):
                        print(e)