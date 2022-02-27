import cv2
import numpy as np
import utils
import pytesseract
import re

file_path = 'snils_example2.jpg'  # INPUT IMG
img = cv2.imread(file_path)

output_img_width = 1050  # PHYSICAL SNILS WIDTH 105мм
output_img_height = 700  # PHYSICAL SNILS HEIGHT 70мм

# DISPLAY CONTOURS
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1)
thres = (30, 30) # MAIN TUNED PARAMETER. MAY VARY FROM SCAN TO SCAN
# thres = (75, 75)
img_threshold = cv2.Canny(img_blurred, thres[0], thres[1])  # APPLY CANNY BLUR
kernel = np.ones((5, 5))
img_threshold = cv2.dilate(img_threshold, kernel, iterations=2)  # APPLY DILATION
img_threshold = cv2.erode(img_threshold, kernel, iterations=1)  # APPLY EROSION

# FIND ALL CONTOURS
img_contours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

# FIND THE BIGGEST CONTOUR
biggest, maxArea = utils.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
if biggest.size != 0:
    biggest = utils.reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
    imgBigContour = utils.drawRectangle(imgBigContour, biggest, 2)
    pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [output_img_width, 0], [0, output_img_height], [output_img_width, output_img_height]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img_gray, matrix, (output_img_width, output_img_height))

    # REMOVE FEW PIXELS FORM EACH SIDE
    img_warped = img_warped[18:img_warped.shape[0] - 18, 18:img_warped.shape[1] - 18]
    img_warped = cv2.resize(img_warped, (output_img_width, output_img_height))

    # APPLY ADAPTIVE THRESHOLD
    # img_warped = cv2.adaptiveThreshold(img_warped, 255, 1, 1, 7, 2)
    # img_warped = cv2.bitwise_not(img_warped)
    # img_warped = cv2.medianBlur(img_warped, 3)

# DISPLAY SINGLE IMG
if False:
    display_img = cv2.resize(img_warped, (output_img_width, output_img_height))
    cv2.imshow('img', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output_img.jpeg', img_warped)

# READ TEXT
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

ru_config = '-l rus'
raw_snils_text = pytesseract.image_to_string(img_warped, config=ru_config)


# PARSE CONTENT
raw_instead_of_parsed = 1
if raw_instead_of_parsed:
    print(raw_snils_text)
else:
    num_regex = '[0-9]\w+\-[0-9]\w+\-[0-9]\w+\-[0-9]\w+'
    snils_num = re.search(num_regex, raw_snils_text).group(0)
    print('№', snils_num)

    surname_regex = '(?<=О\.\s)[А-Я]\w+'
    snils_surname = re.search(surname_regex, raw_snils_text).group(0)
    print('Фамилия', snils_surname)
