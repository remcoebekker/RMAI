import cv2

# We display the test video frames in this window
cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("./video/CX_MC.mp4")

if not cap.isOpened():
    raise IOError("Error opening the video!")

# We count the number of frames...
count = 0
at_end = False
while cap.isOpened() & at_end == False:
    ret, img = cap.read()
    if not ret:
        at_end = True
    else:
        count += 1
#            sequence_number = self.determine_sequence_number(count)
#            cv2.putText(img, "Sequence " + str(sequence_number[0]), (5, 30), self.FONT, 0.6, (255, 255, 0), 2)
#            cv2.putText(img, "Identity " + str(sequence_number[1]), (5, 60), self.FONT, 0.6, (255, 255, 0), 2)
        cv2.putText(img, "Frame count " + str(count), (5, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("win", img)
        key = cv2.waitKey(-1)
        if key == 27:
            at_end = True
        else:
            if key == 32:
                cv2.waitKey(-1)

cap.release()