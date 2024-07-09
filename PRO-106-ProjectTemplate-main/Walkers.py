import cv2


body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')


cap = cv2.VideoCapture('walking.avi')


while True:
    
    ret, frame = cap.read()

    # Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Draw bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(f"Body found at x={x}, y={y}, w={w}, h={h}")

    # Display the output
    cv2.imshow('Body Detection', frame)

    if cv2.waitKey(1) == 32:  # 32 is the Space Key
        break
