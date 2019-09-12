import numpy as np
import cv2 as cv
import random
import os


# our folder path containing some background images
folder_bg = 'BackGround'

# base name for the output images
Base_Name = 'Lizard'

# Output Resolution
Output_Resolution = 400

# output folder for images
folder_image = 'Images/Lizard'
#folder_image = 'Images/Paper'
#folder_image = 'Images/Rock'
#folder_image = 'Images/Scissor'
#folder_image = 'Images/Spock'


# our folder path to videos
folder_videos = 'Videos/Lizard'
#folder_videos = 'Videos/Paper'
#folder_videos = 'Videos/Rock'
#folder_videos = 'Videos/Scissor'
#folder_videos = 'Videos/Spock'

total = 0


# loop on all Background files of the folder and build a list of files paths
BGimages = [os.path.join(folder_bg, f) for f in os.listdir(folder_bg) if os.path.isfile(os.path.join(folder_bg, f))]


# loop on all videos files of the folder and build a list of files paths
videos = [os.path.join(folder_videos, f) for f in os.listdir(folder_videos) if os.path.isfile(os.path.join(folder_videos, f))]


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

# show the list of videos
print(videos)
total = 0

# loop on hand gesture videos
for video in videos:
    cap = cv.VideoCapture(video)

    while(1):

        framebg = cv.imread(np.random.choice(BGimages))

        # Take each frame
        ret, frame = cap.read()
        # if the video end break loop to take the next video
        if ret is False:
            break
        # rezise to a quarter to match with background image
        frame = cv.resize(frame, None, fx=.4, fy=.4, interpolation=cv.INTER_AREA)

        # crop to square output
        crop1 = int((frame.shape[1] - frame.shape[0]) / 2)
        crop2 = crop1 + frame.shape[0]

        frame = frame[:, crop1:crop2]
        framebg = framebg[:, crop1:crop2]

        # Filter to select the masck value easy
        framefilter = cv.bilateralFilter(frame, 6, 75, 75)
        # Convert BGR to HSV
        hsv = cv.cvtColor(framefilter, cv.COLOR_BGR2HSV)

        # get the track bar values
        Hue_L = cv.getTrackbarPos('Hue_L', 'res')
        Saturation_L = cv.getTrackbarPos('Saturation_L', 'res')
        Value_L = cv.getTrackbarPos('Value_L', 'res')
        Hue_H = cv.getTrackbarPos('Hue_H', 'res')
        Saturation_H = cv.getTrackbarPos('Saturation_H', 'res')
        Value_H = cv.getTrackbarPos('Value_H', 'res')

        # define range of green color in HSV
        lower_green = np.array([Hue_L, Saturation_L, Value_L])
        upper_green = np.array([Hue_H, Saturation_H, Value_H])

        # Threshold the HSV image to get only green colors
        mask = cv.inRange(hsv, lower_green, upper_green)
        maskinv = 255 - mask

        # Bitwise-AND mask and original image
        resfg = cv.bitwise_and(frame, frame, mask=maskinv)
        resbg = cv.bitwise_and(framebg, framebg, mask=mask)

        # add the two mask images
        res = cv.add(resfg, resbg)

        # for plot whe need the mask to be in HSV like the image
        mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        mask_3 = cv.cvtColor(mask_3, cv.COLOR_BGR2HSV)

        # put the two image in one single image
        out = np.concatenate((res, mask_3), axis=1)

        # show
        cv.imshow('res', out)

        # press q to abort
        k = cv.waitKey(150)
        if k == ord('q'):
            break

    # print the values of mask
    print(Hue_L, Saturation_L, Value_L, Hue_H, Saturation_H, Value_H)

    print("saving images: ....")

    # define range of green color in HSV
    lower_green = np.array([Hue_L, Saturation_L, Value_L])
    upper_green = np.array([Hue_H, Saturation_H, Value_H])

    while(1):

        framebg = cv.imread(np.random.choice(BGimages))

        # Take each frame
        ret, frame = cap.read()
        if ret is False:
            break
        # rezise to a quarter
        frame = cv.resize(frame, None, fx=.4, fy=.4, interpolation=cv.INTER_AREA)
        # crop to square output
        crop1 = int((frame.shape[1] - frame.shape[0]) / 2)
        crop2 = crop1 + frame.shape[0]

        frame = frame[:, crop1:crop2]
        framebg = framebg[:, crop1:crop2]

        # Filter to select the masck value easy
        framefilter = cv.bilateralFilter(frame, 6, 75, 75)

        # Convert BGR to HSV
        hsv = cv.cvtColor(framefilter, cv.COLOR_BGR2HSV)

        # Threshold the HSV image to get only green colors
        mask = cv.inRange(hsv, lower_green, upper_green)
        maskinv = 255 - mask

        # Bitwise-AND mask and original image
        resfg = cv.bitwise_and(frame, frame, mask=maskinv)
        resbg = cv.bitwise_and(framebg, framebg, mask=mask)

        # add the two mask images
        res = cv.add(resfg, resbg)
        # resize to output resolution
        res = cv.resize(res, (Output_Resolution, Output_Resolution), interpolation=cv.INTER_AREA)

        # show
        cv.imshow('saving...', res)

        # press q to abort
        k = cv.waitKey(1)
        if k == ord('q'):
            break

        # generate the file path and name
        p = os.path.sep.join([folder_image, "{}{}.png".format(
            str(Base_Name), str(total).zfill(5))])

        # save the image
        cv.imwrite(p, res)

        total += 1

print(total)
cv.destroyAllWindows()
