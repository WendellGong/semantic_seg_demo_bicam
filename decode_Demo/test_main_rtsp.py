import cv2

cap = cv2.VideoCapture("rtsp://admin:hkzlbf111@192.168.30.64/Streaming/Channels/1")
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()