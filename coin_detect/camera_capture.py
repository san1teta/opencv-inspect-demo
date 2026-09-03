import cv2

def start_camera_loop(process_func):
    cap = cv2.VideoCapture(0) 
    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break
        processed_frame = process_func(frame)
        cv2.imshow('real_time', processed_frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 
