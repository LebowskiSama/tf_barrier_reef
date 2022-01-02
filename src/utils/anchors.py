def to_yolo_anchor(anchor, im_h=720, im_w=1280):
    x, y, w, h = anchor
    # Normalize to Yolo format
    xnew = x / im_w
    ynew = y / im_h
    wnew = (x + w) / im_w
    hnew = (h + w) / im_h

    return xnew, ynew, wnew, hnew