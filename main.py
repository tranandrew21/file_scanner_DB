import cv2
import imutils
from skimage.filters import threshold_local
from transform import perspective_transform 
from PIL import Image

oImg = cv2.imread('sample.jpg')
copy = oImg.copy()

ratio = oImg.shape[0] / 500
oImgResize = imutils.resize(oImg, height=500)

# Display Output
# cv2.imshow('Resized Image', oImgResize)
# cv2.waitKey(1)

# Filtering the image to greyscale
greyImg = cv2.cvtColor(oImgResize, cv2.COLOR_BGR2GRAY)
# Display Output 
# cv2.imshow('Greyed Image', greyImg)
# cv2.waitKey(0)

# Edge Detector 
blur = cv2.GaussianBlur(greyImg, (5, 5) , 0)
edgeImg = cv2.Canny(blur, 75, 200)

# Display Output
# cv2.imshow('Image edges', edgeImg)
# cv2.waitKey(0)

# Finding the largest contour 
cnts, _ = cv2.findContours(edgeImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) [:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        doc = approx
        break

p = []

for d in doc:
    tuple_point = tuple(d[0])
    cv2.circle(oImgResize, tuple_point, 3, (0, 0, 255), 4)
    p.append(tuple_point)

# Display
# cv2.imshow('Circled Corner Points', oImgResize)
# cv2.waitKey(0)

warpedImg = perspective_transform(copy, doc.reshape(4, 2) * ratio)
warpedImg = cv2.cvtColor(warpedImg, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Warped Image", imutils.resize(warped_image, height=650))
# cv2.waitKey(0)

T = threshold_local(warpedImg, 11, offset=10, method='gaussian')
warped = (warpedImg > T).astype("uint8") * 255
file = cv2.imwrite('./' + 'scan' + '.png', warped)
file = cv2.imwrite('./' + '' + '.png', warped)

# cv2.imshow("Final Scanned image", imutils.resize(warped, height=650))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Convert png to pdf

img = Image.open('./scan.png')
im1 = img.convert('RGB')
im1.save('./scan.pdf')