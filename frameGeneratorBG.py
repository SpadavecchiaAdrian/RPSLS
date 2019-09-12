import cv2 as cv
import random
import os

# our folder path containing some background videos
folder_path = 'VideosBG'
# the output folder for the images
folder_image = 'ImagesBG'

total = 0

# loop on all files of the folder and build a list of files paths
videos = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# show the list of videos
print(videos)

# loop on videos
for video in videos:
    cap = cv.VideoCapture(video)

    while(1):
        # Take each frame
        ret, frame = cap.read()
        # if the video end break loop to take the next video
        if ret is False:
            break
        # rezise to a quarter
        frame = cv.resize(frame, None, fx=.4, fy=.4, interpolation=cv.INTER_AREA)

        # create the image path and name
        p = os.path.sep.join([folder_image, "{}.png".format(
            str(total).zfill(5))])
        # save the image in the path
        cv.imwrite(p, frame)
        # count the total image
        total += 1
        # show the image
        cv.imshow('res', frame)
        # press q to abort
        k = cv.waitKey(1)
        if k == ord('q'):
            break


cv.destroyAllWindows()
