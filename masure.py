import cv2
import pandas as pd
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

videocapture = cv2.VideoCapture(0)

def safe_div(x,y):
    if y==0: return 0
    return x/y

def nothing(x):
    pass

def rescale_frame(frame, percent=100): #Crear la ventana del video
    width = int(frame.shape[1]*percent/100)
    height = int(frame.shape[0]*percent/100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

if not videocapture.isOpened():
    print("No se puede abrir el video")
    exit()

windowName="Medidor"

cv2.namedWindow(windowName)

# Sliders para ajustar imagen

cv2.createTrackbar("Threshold", windowName, 75, 255, nothing)
cv2.createTrackbar("Kernel", windowName, 5, 30, nothing)
cv2.createTrackbar("Iterations", windowName, 1, 10, nothing)

showLive=True
while(showLive):

    ret, frame=videocapture.read()
    frame_resize=rescale_frame(frame)
    if not ret:
        print("No se puede capturar el marco.")
        exit()

    thresh = cv2.getTrackbarPos("Threshold", windowName)
    ret,thresh1 = cv2.threshold(frame_resize, thresh, 255, cv2.THRESH_BINARY)

    kern = cv2.getTrackbarPos("Kernel", windowName)
    kernel = np.ones((kern, kern), np.uint8) # núcleo de imagen cuadrada utilizado para la erosión

    itera=cv2.getTrackbarPos("Iterations", windowName)
    dilation = cv2.dilate(thresh1, kernel, iterations=itera)
    erosion = cv2.erode(dilation, kernel, iterations=itera) #refina todos los bordes en la imagen binaria

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

    contours,hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # encontrar contornos con aproximación simple  cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE

    closing = cv2.cvtColor(closing,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(closing, contours, -1, (128,255,0), 1)

    # centrarse solo en el contorno más grande por área
    areas = [] #lista para contener todas las áreas

    for contour in contours:
      ar = cv2.contourArea(contour)
      areas.append(ar)

    max_area = max(areas)
    max_area_index = areas.index(max_area)  # índice del elemento de lista con área más grande

    cnt = contours[max_area_index - 1] # El contorno del área más grande suele ser la ventana de visualización, ¿por qué?

    cv2.drawContours(closing, [cnt], 0, (0,0,255), 1)

    def midpoint(ptA, ptB): 
      return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    #calcular el cuadro delimitador girado del contorno
    orig = frame_resize.copy()
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    
    # ordenar los puntos en el contorno de modo que aparezcan
    # en la parte superior izquierda, superior derecha, inferior derecha e inferior izquierda
    # orden, luego dibuje el contorno del límite girado
    # caja
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

    
    # bucle sobre los puntos originales y dibujarlos
    for (x, y) in box:
      cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    
    # desempaquete el cuadro delimitador ordenado, luego calcule el punto medio
    # entre las coordenadas superior izquierda y superior derecha, seguido de
    # el punto medio entre las coordenadas inferior izquierda e inferior derecha
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # calcular el punto medio entre los puntos superior izquierdo y superior derecho,
    # seguido del punto medio entre la esquina superior derecha y la esquina inferior derecha
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    
# dibujar los puntos medios en la imagen
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # dibujar líneas entre los puntos medios

    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 1)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 1)
    cv2.drawContours(orig, [cnt], 0, (0,0,255), 1)

    # calcular la distancia euclidiana entre los puntos medios
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # calcular el tamaño del objeto
    P2M4x = 1.2
    P2M10x = 3.2
    P2M20x = 6
    pixelsPerMetric = P2M10x# Conversión de píxeles a micras
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    dimensions = [dimA, dimB]

    
# dibujar los tamaños de los objetos en la imagen
    cv2.putText(orig, "{:.1f}cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    
# calcular el centro del contorno
    M = cv2.moments(cnt)
    cX = int(safe_div(M["m10"],M["m00"]))
    cY = int(safe_div(M["m01"],M["m00"]))

    
# dibuja el contorno y el centro de la forma en la imagen
    cv2.circle(orig, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(orig, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow(windowName, orig)
    cv2.imshow('', closing)
    if cv2.waitKey(30)>=0:
        showLive=False

videocapture.release()
cv2.destroyAllWindows()