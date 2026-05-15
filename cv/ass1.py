import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 100, 100])
    upper = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    M = cv2.moments(mask)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
        print(f"X: {cx}, Y: {cy}")
        cv2.putText(frame,
            f"X:{cx} Y:{cy}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
    
    