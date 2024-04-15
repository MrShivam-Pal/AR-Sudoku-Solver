import cv2 
import process

video = cv2.VideoCapture("http://192.0.0.4:8080/video")
solved_old_sudoku = None

while video.isOpened():
    check , frame = video.read()
    if check:
        frame = cv2.resize(frame , (1080,720))
        sudoku_game= process.main(frame , solved_old_sudoku)
        cv2.imshow("AR Sudoku Solver" , sudoku_game)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  

video.release()
cv2.destroyAllWindows()


