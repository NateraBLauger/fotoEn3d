"""Imágenes de prueba generadas por código (sin archivos externos)."""
import cv2
import numpy as np


def ilustracion_plana(ancho: int = 720, alto: int = 600) -> np.ndarray:
    """Ilustración plana, sin luces ni sombras: pared, piso, dos nubes, dos tajadas de sandía y una naranja."""
    img = np.zeros((alto, ancho, 3), np.uint8)
    img[:] = (205, 200, 120)
    img[int(alto * .62):] = (110, 110, 245)
    for cx, cy, s in [(190, 120, 1.0), (560, 170, 0.75)]:
        for dx, dy, r in [(0, 0, 55), (-55, 18, 40), (55, 18, 42), (0, 28, 45)]:
            cv2.circle(img, (int(cx + dx * s), int(cy + dy * s)), int(r * s), (255, 255, 255), -1)

    def tajada(x, y, w, h, semillas):
        pts = np.array([[x, y + h], [x + w, y + h], [x + w // 2, y]], np.int32)
        cv2.fillConvexPoly(img, pts, (85, 85, 235))
        cv2.rectangle(img, (x, y + h - int(h * .1)), (x + w, y + h), (80, 235, 190), -1)
        for sx, sy in semillas:
            cv2.ellipse(img, (x + sx, y + sy), (6, 10), 0, 0, 360, (30, 30, 40), -1)

    tajada(300, 220, 210, 270, [(105, 90), (80, 140), (135, 150), (105, 200), (70, 205), (145, 210)])
    tajada(190, 330, 130, 160, [(65, 70), (45, 110), (85, 120)])
    cv2.circle(img, (600, 470), 62, (60, 190, 250), -1)
    cv2.circle(img, (585, 450), 8, (30, 140, 220), -1)
    return img
