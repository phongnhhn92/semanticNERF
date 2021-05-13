from PIL import Image
import numpy as np

gt_img = '/home/phong/GT.png'
pred_img = '/home/phong/pred.png'

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"))
    img.save( outfilename )

gt = load_image(gt_img)
H,W,C = gt.shape
loop = W // H
img_name = ''
L = np.split(gt,loop,axis=1)
for i in range(loop):
    if i == 0: img_name = 'source_rgb'
    elif i == 1: img_name = 'source_semantic'
    elif i == 2:
        img_name = 'target_rgb'
    elif i == 3:
        img_name = 'target_semantic'
    save_image(L[i],'./'+img_name+'.jpg')

pred = load_image(pred_img)
H,W,C = pred.shape
loop = W // H
img_name = ''
L = np.split(pred,loop,axis=1)
for i in range(loop):
    if i == 0: img_name = 'pred_target_rgb'
    elif i == 1: img_name = 'pred_target_semantic'
    elif i == 2:
        img_name = 'pred_target_disp'
    elif i == 3:
        img_name = 'pred_target_depth'
    save_image(L[i],'./'+img_name+'.jpg')
print('END')

