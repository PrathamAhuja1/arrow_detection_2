import cv2
import numpy as np


image = cv2.imread('right.jpg')

def create_mask(image):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    return mask


#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(image,(9,9),0)

arrow_mask = create_mask(blurred_image)


contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = np.zeros_like(blurred_image)

cv2.drawContours(blurred_image, contours, -1, (0,255,0), 2)
cv2.drawContours(contour_image, contours, -1, (0,255,0), 2)


largest_contour = max(contours, key=cv2.contourArea)


rect = cv2.minAreaRect(largest_contour)

print(rect)

angle = rect[2]

print(f"Angle of the arrow with respect to horizontal: {angle} degrees")

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    cv2.rectangle(blurred_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.rectangle(contour_image,(x, y), (x + w, y + h), (255, 0, 0), 2)


cv2.imshow('original image', image)
cv2.imshow('Mask', arrow_mask)
cv2.imshow('Blurred_image',blurred_image)
cv2.imshow('Detection',contour_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
