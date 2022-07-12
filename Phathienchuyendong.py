import cv2

cap =cv2.VideoCapture(0) #mở camera của laptop
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #xét chiều cao của frame là 480px
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #xét chiều rộng của frame là 640px

#lấy background cho frame
for i in range(10): #đọc frame liên tục 10 lần
    _, frame = cap.read()
frame = cv2.resize(frame, (640, 480)) #đưa frame về đúng với kích thước đã đặt ban đầu
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (25,25), 0)
frame_truoc = gray  #lấy frame trước

while True:
    _, frame = cap.read()
    #xử lý frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25,25), 0)
    #trừ frame
    abs_image = cv2.absdiff(frame_truoc, gray) #trừ đi sự khác nhau giữa 2 frame
    #ví dụ: 0 - 1 = -1 sẽ gây ra hiện tượng tràn số (nhiễu ánh sáng) nhưng sử dụng abs thì sẽ là 1
    frame_truoc = gray
    _, img_mask = cv2.threshold(abs_image, 30, 255, cv2.THRESH_BINARY) #lọc nhiễu

    #phát hiện chuyển động
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 900: #lọc những contour nhỏ không cần thiết
            continue

        x, y, w, h = cv2.boundingRect(contour) #vẽ hình cn xung quan contour
        cv2.rectangle(frame, (x, y), (x+w, x+h), (0, 255, 0), 3) #frame, chiều cao, chiều dài, màu, dộ dày nét vẽ

    cv2.imshow("phathien", frame) #hiển thị lên video có tên phathien
    #cv2.imshow("phathien", img_mask)

    if cv2.waitKey(1) == ord('q'): #tắt camera với deley 1ms, ấn q sẽ kết thúc
        break