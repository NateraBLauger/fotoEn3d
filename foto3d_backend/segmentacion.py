"""Etapa 1: separar el sujeto del fondo (máscara y canal alfa suave)."""
from __future__ import annotations

import cv2
import numpy as np


def filtro_guiado(guia: np.ndarray, src: np.ndarray, radio: int, eps: float) -> np.ndarray:
    """Filtro guiado (He et al.): suaviza `src` respetando los bordes de `guia`."""
    k = (2 * radio + 1, 2 * radio + 1)

    def media(x):
        return cv2.boxFilter(x, -1, k)

    m_g, m_s = media(guia), media(src)
    a = (media(guia * src) - m_g * m_s) / (media(guia * guia) - m_g * m_g + eps)
    b = m_s - a * m_g
    return media(a) * guia + media(b)


def _componentes_relevantes(m: np.ndarray, fraccion: float = 0.05) -> np.ndarray:
    n, etiq, est, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    areas = est[1:, cv2.CC_STAT_AREA]
    buenos = [i + 1 for i, a in enumerate(areas) if a >= fraccion * areas.max()]
    return np.where(np.isin(etiq, buenos), 255, 0).astype(np.uint8)


def _rellenar_huecos_chicos(m: np.ndarray, fraccion: float = 0.01) -> np.ndarray:
    contornos, jerarquia = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is None:
        return m
    total = float((m > 0).sum())
    salida = m.copy()
    for i, c in enumerate(contornos):
        if jerarquia[0][i][3] != -1 and cv2.contourArea(c) < fraccion * total:
            cv2.drawContours(salida, [c], -1, 255, -1)
    return salida


def _distancia_al_fondo(peq: np.ndarray, b: int, quitar_sombras: bool = False) -> tuple[np.ndarray, float]:
    """Qué tan distinto de "fondo" es cada píxel, modelando el fondo desde el borde de la foto.

    Se combinan dos modelos y se toma el más parecido al fondo:
      * un ajuste polinómico de 2.º grado por canal (sirve para degradados suaves), y
      * los colores dominantes del borde con k-means (sirve para fondos de varios colores planos).
    Devuelve (distancia por píxel, umbral sugerido).
    """
    hp, wp = peq.shape[:2]
    en_borde = np.zeros((hp, wp), bool)
    en_borde[:b], en_borde[-b:], en_borde[:, :b], en_borde[:, -b:] = True, True, True, True
    img = peq.astype(np.float32)

    yy, xx = np.mgrid[0:hp, 0:wp].astype(np.float32)
    x, y = xx / wp - 0.5, yy / hp - 0.5
    base = np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], axis=-1)
    A, C = base[en_borde], img[en_borde]
    usar = np.ones(len(A), bool)
    for _ in range(3):                                   # ajuste robusto: se descartan atípicos
        coef = np.linalg.lstsq(A[usar], C[usar], rcond=None)[0]
        resid = np.linalg.norm(A @ coef - C, axis=1)
        usar = resid <= max(3 * np.median(resid), 8.0)
    dist_poly = np.linalg.norm(base @ coef - img, axis=2)

    muestra = C if len(C) <= 4000 else C[np.random.default_rng(0).choice(len(C), 4000, replace=False)]
    cv2.setRNGSeed(0)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centros = cv2.kmeans(muestra.astype(np.float32), 3, None, criterio, 2, cv2.KMEANS_PP_CENTERS)
    dist_cl = np.min(np.stack([np.linalg.norm(img - c, axis=2) for c in centros]), axis=0)

    dist = np.minimum(dist_poly, dist_cl)
    if quitar_sombras:
        # Una sombra es el mismo color del fondo pero más oscuro: queda sobre el rayo que va del negro al color.
        rayos = []
        for c in centros:
            a = np.clip((img @ c) / (float(c @ c) + 1e-6), 0.45, 1.0)
            rayos.append(np.linalg.norm(img - a[..., None] * c, axis=2))
        dist = np.minimum(dist, np.min(rayos, axis=0))
    dispersion = float(np.median(dist[en_borde]))
    return dist, max(14.0, 3.0 * dispersion)


def mascara_auto(img_bgr: np.ndarray, iteraciones: int = 5, quitar_sombras: bool = False) -> tuple[np.ndarray, list[str]]:
    """Detecta al sujeto con GrabCut inicializado desde el borde de la imagen.

    Supone que el borde de la foto es fondo. Devuelve (máscara uint8 0/255, advertencias).
    """
    alto, ancho = img_bgr.shape[:2]
    esc = min(1.0, 480 / max(alto, ancho))
    peq = img_bgr if esc == 1.0 else cv2.resize(img_bgr, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    hp, wp = peq.shape[:2]
    b = max(2, int(0.03 * min(hp, wp)))

    dist, umbral = _distancia_al_fondo(peq, b, quitar_sombras)

    m = np.full((hp, wp), cv2.GC_PR_FGD, np.uint8)
    m[dist < umbral] = cv2.GC_PR_BGD
    m[dist > 3.0 * umbral] = cv2.GC_FGD
    m[:b], m[-b:], m[:, :b], m[:, -b:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD

    advertencias: list[str] = []
    fg = None
    try:
        cv2.grabCut(peq, m, None, np.zeros((1, 65)), np.zeros((1, 65)), iteraciones, cv2.GC_INIT_WITH_MASK)
        fg = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    except cv2.error:
        pass

    if fg is None or not 0.01 <= fg.mean() <= 0.95:
        fg = dist >= umbral                        # plan B: diferencia de color con el fondo
        advertencias.append("GrabCut no separó bien al sujeto; se usó la diferencia de color con el fondo.")
        if not 0.01 <= fg.mean() <= 0.95:
            fg = np.zeros((hp, wp), bool)           # plan C: elipse central
            cv2.ellipse(fg.view(np.uint8), (wp // 2, hp // 2), (int(wp * .4), int(hp * .4)), 0, 0, 360, 1, -1)
            advertencias.append("No se pudo separar al sujeto: se usó una elipse central. "
                                "Conviene pasar una máscara propia (mascara_sujeto).")

    mascara = fg.astype(np.uint8) * 255
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mascara = _componentes_relevantes(mascara)
    mascara = _rellenar_huecos_chicos(mascara)
    mascara = cv2.resize(mascara, (ancho, alto), interpolation=cv2.INTER_LINEAR)
    mascara = np.where(mascara > 127, 255, 0).astype(np.uint8)
    return cv2.medianBlur(mascara, 5), advertencias


def alfa_suave(img_bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Convierte la máscara dura en un canal alfa con bordes suaves que siguen los bordes de la foto."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    base = (mascara > 0).astype(np.float32)
    radio = max(3, min(gris.shape) // 150)
    alfa = filtro_guiado(gris, base, radio, 1e-4)
    return np.clip((alfa - 0.5) * 1.6 + 0.5, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------- objetos ---
def _dividir_tocados(comp: np.ndarray, img_bgr: np.ndarray, sensibilidad: float) -> list[np.ndarray]:
    """Separa objetos pegados: marcadores = picos de la transformada de distancia; los cortes
    siguen los bordes reales de color de la foto (watershed sobre la imagen)."""
    c = comp.astype(np.uint8)
    dist = cv2.distanceTransform(c, cv2.DIST_L2, 5)
    if dist.max() < 4:
        return [comp]
    firmes = (dist > sensibilidad * dist.max()).astype(np.uint8)
    n, marc = cv2.connectedComponents(firmes)
    if n <= 2:
        return [comp]
    marcadores = np.zeros(c.shape, np.int32)
    marcadores[~comp] = 1
    for k in range(1, n):
        marcadores[marc == k] = k + 1
    cv2.watershed(cv2.GaussianBlur(img_bgr, (0, 0), 1.5), marcadores)
    partes = []
    for k in range(2, n + 1):
        parte = cv2.dilate((marcadores == k).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        parte &= comp                                    # la línea de la cuenca (-1) se reparte
        if parte.sum() >= 0.03 * comp.sum():
            partes.append(parte)
    return partes or [comp]


def separar_objetos(mascara: np.ndarray, img_bgr: np.ndarray, separar_tocados: bool = True,
                    sensibilidad: float = 0.6, max_objetos: int = 9) -> list[np.ndarray]:
    """Divide la máscara en objetos (uint8 0/255), ordenados del más lejano al más cercano.

    La profundidad se estima por la posición de la base: lo que apoya más abajo en la foto está más cerca.
    """
    binaria = (mascara > 0).astype(np.uint8)
    n, etiq = cv2.connectedComponents(binaria, connectivity=8)
    objetos: list[np.ndarray] = []
    for k in range(1, n):
        comp = etiq == k
        if comp.sum() < 0.002 * binaria.size:
            continue                                      # motas diminutas
        objetos.extend(_dividir_tocados(comp, img_bgr, sensibilidad) if separar_tocados else [comp])
    if not objetos:
        return [mascara]
    objetos.sort(key=lambda m: -int(m.sum()))
    if len(objetos) > max_objetos:                        # el exceso se agrupa en un solo objeto
        resto = np.any(objetos[max_objetos - 1:], axis=0)
        objetos = objetos[:max_objetos - 1] + [resto]
    objetos.sort(key=lambda m: int(np.where(m.any(axis=1))[0].max()))
    return [o.astype(np.uint8) * 255 for o in objetos]
