# Wayne H Nixalo -- 2018-Jan-03 02:38 / 2018-Jan-07 21:47
# common utilities
import cv2
import numpy as np
from PIL import Image
from keras_retinanet.utils.image import preprocess_image, resize_image
import time

def detect(image, threshold=0.5, mode='ss', fname='', model=None, quiet=True):
    """
    Runs RetinaNet detection on image, drawing bounding-boxes around
    detections.

    Mode='SS': SemiSupervised
        Will display top 5 predictions to choose from, and returns the chosen
        bounding box. 'q': Quit. '0' signals reject image for manual labelling.

    Mode='Auto' or 'US': Automatic
        Will return top scoring prediction above threshold.
        Automatically signals for image to be rejected for manual labelling
        if score is below threshold.
    """
    # copy image to draw on
    draw = image.copy()
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for neural network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # detect on image
    _,_, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

    # compute predicted labels & scores
    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

    # correct for image scale
    detections[0, :, :4] /= scale

    # Semi-Supervised Mode
    if mode.lower()=='ss':
        # display top 5 predicted labels
        # for idx, (label, score) in enumerate(zip(predicted_abels[:5], scores[:5])):
        c = (255,0,0)
        n = 5
        for idx, score in enumerate(scores[:5]):
            # get bounding box
            b = detections[0, idx, :4].astype(int)

            # shift color
            c = c_shift(c=c, n=n) if idx > 0 else c

            # draw bounding box & label-index | λ_wd: line-width
            λ_wd = 2 if idx > 0 else 4
            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), c, λ_wd)
            caption = f'{idx+1}'
            cv2.putText(draw, caption, (b[0]+10, b[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(draw, caption, (b[0]+10, b[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 3, cv2.LINE_AA)

            # print label indices, bounding boxes, and scores
            print(f'{idx+1}: {score:.12f} -- {b}')

        # display image boundingbox overlay
        cv2.imshow(fname, draw)
        cv2.waitKey(1)
        time.sleep(1e-2)

        # semi-supervised labelling
        inp = None
        while True:
            inp = input("Index: ")
            if type(inp) == str and inp.isdigit():
                if int(inp) > 0 and int(inp) < 5:
                    inp = int(inp)
                    break
                elif int(inp) == 0:
                    inp = 0
                    break
            # enter letter to reload window
            elif type(inp) == str and not inp.isdigit():
                cv2.destroyAllWindows()
                cv2.imshow(fname, draw)
                cv2.waitKey(1)
                time.sleep(1e-2)

                if inp == 'q':
                    print('Exit Signal Entered: Quitting.')
                    return -1

        # close image window
        cv2.destroyAllWindows()

        # return chosen bounding box OR reject-flag
        if inp == 0:
            if not quiet:
                print(f'{fname} rejected: manual labelling')
            return inp

        b = detections[0, inp-1, :4].astype(int)

    # UnSupervised/Automatic Mode
    # automatically pull highest-confidence bounding box
    else:
        if scores[0] >= threshold:
            b = detections[0, 0, :4].astype(int)
        else:
            return 0

    # convert to ints
    for i in range(len(b)):
        b[i] = max(0, b[i])

    if not quiet:
        print(f'Bbox: {b} chosen')

    return b    # numpy.int64 ndarray


# Wayne Nixalo -- 2018-Jan-03 00:37 - 01:17
# Function to shift colors
# ie: if n=3 --> you get: R, G, B
#        n=2 -->          R, G
#        n=n --> n colors shifted by 255*3/n

def c_shift(c=[255,0,0], n=1, shifts=1, val=255, quiet=True):
    """
    Creates equal-sized shifts in color, where size is val*3/n, starting with
    pure-Red and ending with pure-Blue (assuming RGB).
    """
    f = val*3/n # 255*3/5 = 153
    shifts = min(n, shifts)
    c = list(c)

    for i in range(shifts):
        if not quiet:  print(f'print: {c}')
        if i == n - 1: continue

        # find first nonzero
        idx = next(i for i,x in enumerate(c) if x > 0)

        move = 0

        if c[idx] >= f:
            move += f
            c[idx] -= f
        else:
            move += c[idx]
            c[idx] = 0
        if move == f:
            c[idx+1] += f
        else:
            c[idx+1] += move
            move = f -  move
            if idx+2 <= len(c)-1:
                c[idx+2] += move
                c[idx+1] -= move

    # round all to ints
    c = [round(x) for x in c]

    return tuple(c) # changed to tuple for OpenCV


# Wayne H Nixalo -- 2018-Jan-03 13:00 - 14:00
# image cropper
def crop(image, bbox):
    """
    bbox:[x1,y1,x2,y2]
    roi: [y1:y2,x1:x2]
    image: (ndarray)
    """
    p1,p2 = (bbox[0],bbox[1]), (bbox[2],bbox[3])
    crop = image.copy()
    return crop[p1[1]:p2[1], p1[0]:p2[0]]

# 2018-Jan-07 21:46
# overlay: http://stackoverflow.com/a/14102014/627517
# via: https://sourcegraph.com/github.com/lexfridman/boring-detector@master/-/blob/boring_common.py#L60
def overlay_image(λ_img, s_img, x_offset=0, y_offset=0):
    """
    Overlays foreground image `s_img` atop background image `λ_img` adjusted by
    x_- & y_offset.

    Returns: overlayed image (ndarray)
    """
    assert y_offset + s_img.shape[0] <= λ_img.shape[0]
    assert x_offset + s_img.shape[1] <= λ_img.shape[1]

    λ_img = λ_img.copy()

    # if an alpha-channel exists in fg-image: remove it
    if s_img.shape[-1] == 4:
        for c in range(0, 3):
            λ_img[y_offset:y_offset+s_img.shape[0],
                  x_offset:x_offset+s_img.shape[1], c] = (
                                s_img[:,:,c] * (s_img[:,:,3]/255.) +
                                λ_img[y_offset:y_offset+s_img.shape[0],
                                      x_offset:x_offset+s_img.shape[1], c] *
                                (1. - s_img[:,:,3]/255.))
    else:
        # no alpha-channel:
        λ_img[y_offset:y_offset+s_img.shape[0],
              x_offset:x_offset+s_img.shape[1]] = s_img

    return λ_img



### testing crop():
# fpath = 'data/train/006440-006548/006530.jpg'
# image = Image.open(fpath)
# image = np.array(image)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# bbox = [7,53,168,306]
# cropped = crop(image, bbox)
# cv2.imshow('', cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


### testing c_shift():
# c_shift(c=[150,0,0], n=5, shifts=5, val=150, quiet=False)
# # print: [150, 0, 0]
# # print: [60.0, 90.0, 0]
# # print: [0, 120.0, 30.0]
# # print: [0, 30.0, 120.0]
# # print: [0, 0, 150.0]
# # [0, 0, 150]


# c_shift(c=[150,0,0], n=5, shifts=1, val=150, quiet=False)
# # print: [150, 0, 0]
# # [60, 90, 0]


# c_shift(c=[150,0,0], n=2, shifts=3, val=150, quiet=False)
# # print: [150, 0, 0]
# # print: [0, 75.0, 75.0]
# # [0, 75, 75]


# c_shift(c=[255,0,0], n=3, shifts=2, quiet=False)
# # print: [255, 0, 0]
# # print: [0.0, 255.0, 0]
# # [0, 0, 255]


# c_shift(c=[255,0,0], n=5, shifts=5, quiet=False)
# # print: [255, 0, 0]
# # print: [102.0, 153.0, 0]
# # print: [0, 204.0, 51.0]
# # print: [0, 51.0, 204.0]
# # print: [0, 0, 255.0]
# # [0, 0, 255]

# # type(round(2.009))
# int
