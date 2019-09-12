import cv2 as cv
import random
import os


# Output Resolution
Output_Resolution = 96

# output folder for images

folder_image = 'Images/Lizard'
#folder_image = 'Images/Paper'
#folder_image = 'Images/Rock'
#folder_image = 'Images/Scissor'
#folder_image = 'Images/Spock'


total = 0

# loop on all files of the folder and build a list of files paths
images = [os.path.join(folder_image, f) for f in os.listdir(folder_image) if os.path.isfile(os.path.join(folder_image, f))]

print(len(images))


def nothing(x):
    pass


for image in images:
    cap = cv.imread(image)

    cap = cv.resize(cap, (Output_Resolution, Output_Resolution), interpolation=cv.INTER_AREA)

    cv.imwrite(image, cap)
    total += 1

    cv.imshow('cap', cap)
    k = cv.waitKey(1)
    if k == ord('q'):
        break

print(total)
cv.destroyAllWindows()
