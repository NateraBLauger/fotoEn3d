"""Estilo "arcilla": quitar los contornos del dibujo, aplanar los colores y separarlo en piezas.

En un render 3D de juguete no hay líneas negras ni texturas pintadas: cada pieza es un volumen de color
liso. Estas funciones preparan la imagen para que el volumen y la luz hagan el trabajo.
"""
from __future__ import annotations

import cv2
import numpy as np


def quitar_contornos(img_bgr: np.ndarray, mascara: np.ndarray, fuerza: float = 1.0) -> np.ndarray:
    """Borra las líneas oscuras largas (los contornos del dibujo) rellenándolas con el color vecino.

    Se detectan con un black-hat morfológico (líneas oscuras más finas que el núcleo). Solo se borran los
    trazos largos: los detalles cortos (boca, pestañas, semillas) se conservan. También se reconstruye la
    franja de fondo pegada al sujeto para que el relleno use solo colores del interior.
    """
    if fuerza <= 0:
        return img_bgr
    alto, ancho = img_bgr.shape[:2]
    m = mascara > 0
    k = 2 * max(2, int(round(0.012 * max(alto, ancho)))) + 1
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    negro = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    lineas = ((negro > 40) & m).astype(np.uint8)

    ys, xs = np.where(m)
    diagonal = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    n, etiq, est, _ = cv2.connectedComponentsWithStats(lineas, connectivity=8)
    largas = np.zeros_like(lineas)
    for i in range(1, n):
        x, y, w, h, _a = est[i]
        if np.hypot(w, h) > 0.22 * diagonal:
            largas[etiq == i] = 1
    largas = cv2.dilate(largas, np.ones((3, 3), np.uint8))

    franja_fuera = (cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & ~m
    desconocido = ((largas > 0) | franja_fuera).astype(np.uint8) * 255
    limpio = cv2.inpaint(img_bgr, desconocido, 3, cv2.INPAINT_TELEA)
    mezcla = img_bgr.astype(np.float32) * (1 - fuerza) + limpio.astype(np.float32) * fuerza
    return np.clip(mezcla, 0, 255).astype(np.uint8)


def aplanar_color(img_bgr: np.ndarray, fuerza: float = 1.0) -> np.ndarray:
    """Colores lisos por zonas (filtro bilateral repetido + desplazamiento de media), sin perder los bordes."""
    if fuerza <= 0:
        return img_bgr
    a = cv2.bilateralFilter(img_bgr, 9, 35, 9)
    a = cv2.bilateralFilter(a, 9, 35, 9)
    a = cv2.pyrMeanShiftFiltering(a, 6, 14)
    mezcla = img_bgr.astype(np.float32) * (1 - fuerza) + a.astype(np.float32) * fuerza
    return np.clip(mezcla, 0, 255).astype(np.uint8)


def _fusionar_parecidas(partes: np.ndarray, lab: np.ndarray, umbral: float) -> np.ndarray:
    """Une piezas que se tocan y tienen casi el mismo color medio (Lab), de la más parecida a la menos."""
    partes = partes.copy()
    while True:
        ids = [int(i) for i in np.unique(partes) if i > 0]
        if len(ids) < 2:
            break
        medias = {i: lab[partes == i].mean(axis=0) for i in ids}
        pares = set()
        for a, b in ((partes[:, :-1], partes[:, 1:]), (partes[:-1], partes[1:])):
            dif = (a != b) & (a > 0) & (b > 0)
            pares.update(zip(np.minimum(a[dif], b[dif]).tolist(), np.maximum(a[dif], b[dif]).tolist()))
        if not pares:
            break
        mejor = min(pares, key=lambda par: float(np.linalg.norm(medias[par[0]] - medias[par[1]])))
        if float(np.linalg.norm(medias[mejor[0]] - medias[mejor[1]])) >= umbral:
            break
        partes[partes == mejor[1]] = mejor[0]
    return partes


def segmentar_partes(img_bgr: np.ndarray, mascara: np.ndarray, k: int = 8,
                     fraccion_min: float = 0.012, fusion_color: float = 22.0) -> np.ndarray:
    """Divide el sujeto en piezas de color: k-means en Lab + componentes conectados.

    Devuelve un mapa int32 (0 = fondo; 1..N = piezas). Las piezas diminutas se unen a la vecina con la que
    comparten más borde, y las piezas vecinas de color muy parecido (distancia en Lab menor que `fusion_color`)
    se fusionan: así un vaso con reflejos de tonos parecidos sigue siendo una sola pieza.
    """
    m = mascara > 0
    if m.sum() < 50 * max(k, 1):
        return m.astype(np.int32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    datos = lab[m]
    if len(datos) > 40000:
        datos = datos[np.random.default_rng(0).choice(len(datos), 40000, replace=False)]
    cv2.setRNGSeed(0)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centros = cv2.kmeans(datos, k, None, criterio, 3, cv2.KMEANS_PP_CENTERS)
    dist = np.stack([((lab - c) ** 2).sum(axis=2) for c in centros])
    etiq = (dist.argmin(axis=0) + 1).astype(np.uint8)
    etiq[~m] = 0
    etiq = cv2.medianBlur(cv2.medianBlur(etiq, 7), 7)          # alisa los bordes y quita franjas finas
    etiq[~m] = 0

    partes = np.zeros(m.shape, np.int32)
    siguiente = 1
    for l in range(1, k + 1):
        n, comp = cv2.connectedComponents((etiq == l).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            partes[comp == i] = siguiente
            siguiente += 1

    minimo = fraccion_min * float(m.sum())
    nucleo = np.ones((3, 3), np.uint8)
    while True:
        cuentas = np.bincount(partes.ravel())
        chicas = [i for i in range(1, len(cuentas)) if 0 < cuentas[i] < minimo]
        if not chicas or (partes > 0).sum() == 0:
            break
        i = min(chicas, key=lambda j: cuentas[j])
        region = (partes == i).astype(np.uint8)
        anillo = (cv2.dilate(region, nucleo, iterations=2) > 0) & (region == 0) & (partes > 0)
        if anillo.any():
            partes[region > 0] = int(np.bincount(partes[anillo]).argmax())
        else:
            partes[region > 0] = 0
    if fusion_color > 0:
        partes = _fusionar_parecidas(partes, lab, fusion_color)
    ids = np.unique(partes[partes > 0])
    tabla = np.zeros(int(partes.max()) + 1, np.int32)
    tabla[ids] = np.arange(1, len(ids) + 1)
    return tabla[partes]


def colorear_partes(partes: np.ndarray) -> np.ndarray:
    """Visualización de las piezas (BGR uint8), con colores fijos por pieza."""
    rng = np.random.default_rng(5)
    tabla = rng.integers(60, 235, (int(partes.max()) + 1, 3)).astype(np.uint8)
    tabla[0] = 255
    return tabla[partes]


def indice_de_textura(img_bgr: np.ndarray, mascara: np.ndarray) -> float:
    """Fracción de píxeles del sujeto con gradiente intermedio (ni zona lisa ni borde de dibujo).

    Los dibujos de colores lisos dan valores bajos (menos de 0,4); las fotos y los renders con mucho detalle,
    más de 0,5. Sirve para avisar cuando el estilo "piezas" no es la mejor opción.
    """
    esc = min(1.0, 320 / max(img_bgr.shape[:2]))
    g = cv2.cvtColor(cv2.resize(img_bgr, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    m = cv2.resize(mascara, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if not m.any():
        return 0.0
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))[m]
    return float(((mag > 12) & (mag < 80)).mean())


def separar_elementos(partes: np.ndarray, maximo: int = 12) -> list[np.ndarray]:
    """Cada pieza de color es un elemento independiente (máscara uint8 0/255).

    Se ordenan de mayor a menor área: los grandes quedan atrás y los pequeños (ojos, cereza, semillas) por delante,
    de modo que cada elemento proyecta su sombra sobre los que tiene debajo. Si hay más de `maximo`, los más
    chicos se agrupan en uno solo.
    """
    ids = [i for i in np.unique(partes) if i > 0]
    if not ids:
        return [np.zeros(partes.shape, np.uint8)]
    masks = [(partes == i) for i in ids]
    masks.sort(key=lambda m: -int(m.sum()))
    if len(masks) > maximo:
        resto = np.any(masks[maximo - 1:], axis=0)
        masks = masks[:maximo - 1] + [resto]
    return [m.astype(np.uint8) * 255 for m in masks]
