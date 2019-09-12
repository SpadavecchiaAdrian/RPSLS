import numpy as np
import cv2 as cv

cap = cv.VideoCapture('/home/adrian/Desktop/RockScissorsPaperLizardSpock/DataSet/VID_20190829_171730866.mp4')

bkground = cv.VideoCapture('VID_20190830_102028895.mp4')


def nothing(x):
    pass


cv.namedWindow('res')
# create trackbars for color change
cv.createTrackbar('Hue_L', 'res', 0, 179, nothing)
cv.createTrackbar('Saturation_L', 'res', 0, 255, nothing)
cv.createTrackbar('Value_L', 'res', 0, 255, nothing)
cv.createTrackbar('Hue_H', 'res', 0, 179, nothing)
cv.createTrackbar('Saturation_H', 'res', 0, 255, nothing)
cv.createTrackbar('Value_H', 'res', 0, 255, nothing)

# Take each frame
#_, frame = cap.read()
while(1):
    _, framebg = bkground.read()
    framebg = cv.resize(framebg, None, fx=.4, fy=.4, interpolation=cv.INTER_AREA)
    # Take each frame
    _, frame = cap.read()

    # rezise to a quarter
    frame = cv.resize(frame, None, fx=.4, fy=.4, interpolation=cv.INTER_AREA)
    # Filter to select the masck value easy
    framefilter = cv.bilateralFilter(frame, 6, 75, 75)
    # Convert BGR to HSV
    hsv = cv.cvtColor(framefilter, cv.COLOR_BGR2HSV)

    Hue_L = cv.getTrackbarPos('Hue_L', 'res')
    Saturation_L = cv.getTrackbarPos('Saturation_L', 'res')
    Value_L = cv.getTrackbarPos('Value_L', 'res')
    Hue_H = cv.getTrackbarPos('Hue_H', 'res')
    Saturation_H = cv.getTrackbarPos('Saturation_H', 'res')
    Value_H = cv.getTrackbarPos('Value_H', 'res')

    # define range of blue color in HSV
    lower_blue = np.array([Hue_L, Saturation_L, Value_L])
    upper_blue = np.array([Hue_H, Saturation_H, Value_H])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    maskinv = 255 - mask
    # Bitwise-AND mask and original image
    resfg = cv.bitwise_and(frame, frame, mask=mask)
    resbg = cv.bitwise_and(framebg, framebg, mask=maskinv)
    res = cv.add(resfg, resbg)
    #cv.imshow('frame', frame)
    #cv.imshow('mask', mask)
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    mask_3 = cv.cvtColor(mask_3, cv.COLOR_BGR2HSV)
    out = np.concatenate((res, mask_3), axis=1)
    cv.imshow('res', out)
    k = cv.waitKey(30)
    if k == ord('q'):
        break
    if k == ord('s'):
        cv.imwrite('hand.png', res)

    #k = cv.waitKey(250) & 0xFF
    # if k == 27:
    #    break
    # else:
    #    cv.imwrite('hand.png', res)
print(Hue_L, Saturation_L, Value_L, Hue_H, Saturation_H, Value_H)
cv.destroyAllWindows()
