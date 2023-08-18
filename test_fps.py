import cv2
import time

def test_camera_fps(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    frame_count = 0
    start_time = time.time()
    avg_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        frame_count += 1

        if frame_count %16 == 0:
            elapsed_time = time.time() - start_time
            avg_fps = (frame_count / elapsed_time)
            start_time = time.time()
            frame_count = 0

        cv2.putText(frame, "Average FPS: {:.2f}".format(avg_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) >27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_fps()
