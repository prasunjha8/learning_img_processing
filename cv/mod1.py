import cv2
import numpy as np

cap = cv2.VideoCapture(0)# 0 hota hai default webcam ke liye, agar aapke paas multiple cameras hain toh aap 1, 2, etc. try kar sakte hain

while True:
    ret, frame = cap.read() # ret boolean value return karta hai jo batata hai ki frame successfully read hua ya nahi, aur frame actual image data hota hai
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([35, 100, 100])   # adjust for green
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()