from config import IMAGE_SIZE
from bbox_utils import BoundingBox

def to_yolo_anchor(box):
    x, y, w, h = box
    im_w, im_h = IMAGE_SIZE
    # Normalize to Yolo format
    xnew = float(x / im_w)
    ynew = float(y / im_h)
    wnew = float((x + w) / im_w)
    hnew = float((h + w) / im_h)

    return xnew, ynew, wnew, hnew