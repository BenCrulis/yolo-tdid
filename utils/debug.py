from ultralytics.utils.plotting import Annotator


def label_img_with_box(img, box, label=""):
    annotator = Annotator(img, pil=True)
    annotator.box_label(box, label=label, color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False) 
    return annotator


