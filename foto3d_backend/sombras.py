"""Etapa 4: sombras y grosor (extrusión)."""
from __future__ import annotations

import cv2
import numpy as np


def trasladar(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    alto, ancho = img.shape[:2]
    m = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, m, (ancho, alto), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def sombra_proyectada(alfa: np.ndarray, dx: float, dy: float, sigma: float) -> np.ndarray:
    s = trasladar(alfa, dx, dy)
    return cv2.GaussianBlur(s, (0, 0), max(0.1, sigma)) if sigma > 0 else s


def sombra_total(alfa: np.ndarray, dx: float, dy: float, sigma: float,
                 opacidad: float, contacto: float) -> np.ndarray:
    """Sombra suave y desplazada + sombra de contacto más oscura y pegada al objeto."""
    lejana = sombra_proyectada(alfa, dx, dy, sigma) * opacidad
    cerca = sombra_proyectada(alfa, dx * 0.3, dy * 0.3, sigma / 3) * contacto
    return 1.0 - (1.0 - lejana) * (1.0 - cerca)


def extruir(color: np.ndarray, alfa: np.ndarray, dx: float, dy: float, pasos: int,
            oscurecer: float) -> tuple[np.ndarray, np.ndarray]:
    """Grosor: copias desplazadas del sujeto, de la más lejana a la más cercana.

    Devuelve (color oscurecido float32 [0,1], alfa de la pared). El sujeto se dibuja luego por encima.
    """
    acum_c = np.zeros_like(color)
    acum_a = np.zeros_like(alfa)
    for k in range(pasos, 0, -1):
        f = k / pasos
        sc = trasladar(color, dx * f, dy * f)
        sa = trasladar(alfa, dx * f, dy * f)
        acum_c = acum_c * (1 - sa[..., None]) + sc * sa[..., None]
        acum_a = acum_a + sa * (1 - acum_a)
    return acum_c * oscurecer, acum_a


def sombra_piso(alfa: np.ndarray, base_y: float, inclinacion: float, largo: float,
                sigma: float, opacidad: float) -> np.ndarray:
    """Sombra del objeto proyectada sobre el piso.

    La silueta se refleja respecto de la línea de base, se aplana (`largo`) y se inclina
    (`inclinacion` > 0: la sombra cae hacia la derecha). Cerca de la base es nítida y se va
    desenfocando al alejarse ("endurecimiento por contacto"), como en una sombra real.
    """
    alto, ancho = alfa.shape
    m = np.array([[1, -inclinacion, inclinacion * base_y],
                  [0, -largo, (1 + largo) * base_y]], np.float32)
    s = cv2.warpAffine(alfa, m, (ancho, alto), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    s = s * (np.arange(alto, dtype=np.float32)[:, None] >= base_y - 1)       # solo por debajo de la base
    cerca = cv2.GaussianBlur(s, (0, 0), max(0.1, sigma * 0.35))
    lejos = cv2.GaussianBlur(s, (0, 0), max(0.1, sigma * 1.6))
    filas = alfa.any(axis=1)
    alto_obj = max(1.0, float(np.where(filas)[0].max() - np.where(filas)[0].min())) if filas.any() else 1.0
    t = np.clip((np.arange(alto, dtype=np.float32)[:, None] - base_y) / (largo * alto_obj), 0, 1)
    return (cerca * (1 - t) + lejos * t) * opacidad
