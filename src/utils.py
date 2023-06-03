import cv2 
import numpy as np 
def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]
    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale
def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size

class CropImage:
    def calculate_new_box(self,source_width, source_height, bbox, scale):
        x_coord              = bbox[0]
        y_coord              = bbox[1]
        box_width            = bbox[2]
        box_height           = bbox[3]

        scale                = min((source_height-1)/box_height, min((source_width-1)/box_width, scale))

        new_width            = box_width * scale
        new_height           = box_height * scale
        center_x, center_y   = box_width/2+x_coord, box_height/2+y_coord

        left_top_x           = center_x-new_width/2
        left_top_y           = center_y-new_height/2
        right_bottom_x       = center_x+new_width/2
        right_bottom_y       = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x  -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y  -= left_top_y
            left_top_y = 0

        if right_bottom_x > source_width-1:
            left_top_x      -= right_bottom_x-source_width+1
            right_bottom_x   = source_width-1

        if right_bottom_y > source_height-1:
            left_top_y      -= right_bottom_y-source_height+1
            right_bottom_y   = source_height-1

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img          = cv2.resize(org_img, (out_w, out_h))
        else:
            source_height, source_width, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self.calculate_new_box(source_width, source_height, bbox, scale)

            img = org_img[left_top_y: right_bottom_y+1,
                          left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img
