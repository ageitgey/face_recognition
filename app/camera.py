# created by Jackson Kim on 2026-05

import cv2
from app.recognition import recognize_faces


def run_webcam():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        results = recognize_faces(frame)

        for result in results:
            top = result["location"]["top"]
            right = result["location"]["right"]
            bottom = result["location"]["bottom"]
            left = result["location"]["left"]
            name = result["name"]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()