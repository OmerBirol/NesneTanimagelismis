import cv2
import cvzone
import time

# Nesne algılama eşiği
thres = 0.55
nmsThres = 0.2

# Kamera başlat
cap = cv2.VideoCapture(0)  # Bilgisayar kamerası kullanılıyor
cap.set(3, 640)  # Genişlik
cap.set(4, 480)  # Yükseklik

# Video dosyasını aç
# video_path = "20250209-1932-7381769.mp4"  # Buraya test etmek istediğin video dosyasının adını yaz
# cap = cv2.VideoCapture(video_path)

# COCO sınıflarını yükle
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().strip().split('\n')

# Modeli yükle
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# FPS hesaplama için başlangıç zamanı
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Video bitti veya hata oluştu.")
        break  # Video sona erdiğinde çık

    # FPS hesapla
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

    # Son algılanan nesnenin koordinatlarını sakla
    last_x, last_y, last_w, last_h = 0, 0, 0, 0

    if len(classIds) > 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x, y, w, h = box
            last_x, last_y, last_w, last_h = x, y, w, h  # Son koordinatları güncelle
            cvzone.cornerRect(img, box)  # Algılanan nesneye çerçeve ekle
            
            # Nesne adı ve güven oranı
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}%',
                        (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)

    # Koordinat bilgilerini sol alt köşeye yerleştir
    height, width = img.shape[:2]
    
    # Arka plan dikdörtgeni çiz
    padding = 10
    text_height = 25
    rect_height = 125  # Adjusted to fit FPS
    rect_width = 200
    
    # Yarı saydam siyah arka plan
    overlay = img.copy()
    cv2.rectangle(overlay, (padding, height - rect_height - padding), 
                 (rect_width + padding, height - padding), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Koordinat bilgilerini yaz
    cv2.putText(img, f'X: {last_x}, Y: {last_y}',
                (padding + 5, height - rect_height + text_height), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'W: {last_w}, H: {last_h}',
                (padding + 5, height - rect_height + text_height * 2), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'Object: {classNames[classIds[0]-1] if len(classIds) > 0 else "None"}',
                (padding + 5, height - rect_height + text_height * 3), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 2)

    # FPS değerini ekrana yaz
    cv2.putText(img, f'FPS: {int(fps)}',
                (padding + 5, height - rect_height + text_height * 4), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 2)

    cv2.imshow("Detection", img)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # 'q' tuşuna basınca çık
        break

cap.release()
cv2.destroyAllWindows()
