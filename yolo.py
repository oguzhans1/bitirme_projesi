# gerekli paketleri içe aktarıldı
import numpy as np
import argparse
import time
import cv2
import os

# argüman oluşturuldu ve ayrıştırıldı
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# YOLO modelinin eğitildiği .names sınıfının etiketlerini yükleyin
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# olası her sınıf etiketini temsil etmek için bir renk listesi kullanımı
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# YOLO ağırlıklarına ve model yapılandırmasına giden yolları türetme
weightsPath = os.path.sep.join([args["yolo"], "yolov4_obj.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4_obj.cfg"])

# obj.names veri setinde (1 sınıf) eğitilmiş YOLO nesne dedektörümüzü yükleyin
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# giriş resmimizi yükleyin ve uzamsal boyutlarını alın
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# YOLO'dan yalnızca ihtiyacımız olan *çıktı* katman adlarını belirleyin
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# giriş görüntüsünden bir blob oluşturun ve ardından bir ileri işlem gerçekleştirin
# bize sınırlayıcı kutularımızı veren YOLO nesne dedektörünün geçişi ve
# ilişkili olasılıklar
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# YOLO'da zamanlama bilgilerini göster
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# algılanan sınırlayıcı kutular, gizlilikler ve
# sınıf kimliği, sırasıyla
boxes = []
confidences = []
classIDs = []

# katman çıktılarının her biri üzerinde döngü
for output in layerOutputs:
	# algılamaların her biri üzerinde döngü
	for detection in output:
		# sınıf kimliğini ve güvenini (yani, olasılığını) çıkarın
		# geçerli nesne algılama
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# tespit edilmesini sağlayarak zayıf tahminleri filtreleyin
		# olasılık minimum olasılıktan daha büyük
		if confidence > args["confidence"]:
			# sınırlayıcı kutu koordinatlarını
			# görüntünün boyutu, YOLO'nun aslında
			# sınırlamanın merkez (x, y) koordinatlarını döndürür
			# kutunun ardından kutuların genişliği ve yüksekliği
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# tepeyi elde etmek için merkez (x, y) koordinatlarını kullanın ve
			# sınırlayıcı kutunun sol köşesi
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# sınırlayıcı kutu koordinatları, güven listemizi güncelleyin,
			# ve class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# # zayıf, örtüşen sınırlayıcı kutuları bastırmak 
#için maksimum olmayan bastırma uygula

idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# en az bir algılamanın mevcut olduğundan emin olun
if len(idxs) > 0:
	
	for i in idxs.flatten():
		
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# resmin üzerine bir sınırlayıcı kutu dikdörtgeni çizin ve etiketleyin
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# çıktı görüntüsünü göster
cv2.imshow("Image", image)
cv2.waitKey(0)
