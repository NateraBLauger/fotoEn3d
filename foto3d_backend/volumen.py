"""Etapa 2: volumen. Se "infla" la silueta para crear un mapa de alturas y de ahí las normales."""
from __future__ import annotations

import cv2
import numpy as np


def _poisson_inflado(mascara: np.ndarray, lado: int = 192) -> np.ndarray:
    """Resuelve  laplaciano(u) = -1  dentro de la silueta, con u = 0 en el borde y fuera (Jacobi).

    El resultado es una cúpula suave: crece de forma continua desde el borde hasta el centro, sin
    mesetas ni pliegues. Se resuelve de grueso a fino (cada nivel parte de la solución del anterior),
    lo que converge rápido y no requiere librerías extra.
    """
    alto, ancho = mascara.shape
    esc_max = min(1.0, lado / max(alto, ancho))
    niveles = sorted({max(16, int(round(max(alto, ancho) * esc_max * f))) for f in (0.125, 0.25, 0.5, 1.0)})
    nucleo = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32) / 4.0
    u = None
    previo = None
    for n in niveles:
        e = n / max(alto, ancho)
        tam = (max(4, round(ancho * e)), max(4, round(alto * e)))
        m = (cv2.resize((mascara > 0).astype(np.float32), tam, interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32)
        if u is None:
            u = np.zeros(m.shape, np.float32)
        else:                                            # la solución crece con el cuadrado de la escala
            u = cv2.resize(u, tam, interpolation=cv2.INTER_LINEAR) * (n / previo) ** 2
        for _ in range(250 if n < niveles[-1] else 150):
            u = (cv2.filter2D(u, -1, nucleo, borderType=cv2.BORDER_CONSTANT) + 0.25) * m
        previo = n
    return cv2.resize(u, (ancho, alto), interpolation=cv2.INTER_CUBIC)


def _poisson_partes(partes: np.ndarray, lado: int = 192) -> np.ndarray:
    """Inflado de Poisson de cada pieza por separado (u = 0 en el borde de cada una).

    Dos píxeles solo se "ven" si pertenecen a la misma pieza, así que entre piezas vecinas queda un surco
    y cada pieza se abomba como un cojín. Igual que `_poisson_inflado`, se resuelve de grueso a fino.
    """
    alto, ancho = partes.shape
    esc_max = min(1.0, lado / max(alto, ancho))
    niveles = sorted({max(16, int(round(max(alto, ancho) * esc_max * f))) for f in (0.125, 0.25, 0.5, 1.0)})

    def vecino(a, dy, dx, relleno):
        r = np.full_like(a, relleno)
        ys, yd = (slice(0, a.shape[0] - dy), slice(dy, None)) if dy >= 0 else (slice(-dy, None), slice(0, a.shape[0] + dy))
        xs, xd = (slice(0, a.shape[1] - dx), slice(dx, None)) if dx >= 0 else (slice(-dx, None), slice(0, a.shape[1] + dx))
        r[yd, xd] = a[ys, xs]
        return r

    u, previo = None, None
    for n in niveles:
        e = n / max(alto, ancho)
        tam = (max(4, round(ancho * e)), max(4, round(alto * e)))
        etiq = cv2.resize(partes.astype(np.float32), tam, interpolation=cv2.INTER_NEAREST).astype(np.int32)
        m = (etiq > 0).astype(np.float32)
        iguales = [((etiq == vecino(etiq, dy, dx, -1)) & (etiq > 0)).astype(np.float32)
                   for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        u = np.zeros(m.shape, np.float32) if u is None else cv2.resize(u, tam, interpolation=cv2.INTER_LINEAR) * (n / previo) ** 2 * m
        for _ in range(250 if n < niveles[-1] else 150):
            suma = sum(ig * vecino(u, dy, dx, 0.0) for ig, (dy, dx) in zip(iguales, ((1, 0), (-1, 0), (0, 1), (0, -1))))
            u = (suma + 1.0) * 0.25 * m
        previo = n
    u = cv2.resize(u, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    borde = np.zeros((alto, ancho), bool)                     # el surco entre piezas queda en cero
    borde[1:] |= partes[1:] != partes[:-1]
    borde[:-1] |= partes[1:] != partes[:-1]
    borde[:, 1:] |= partes[:, 1:] != partes[:, :-1]
    borde[:, :-1] |= partes[:, 1:] != partes[:, :-1]
    u[borde] = 0.0
    return np.clip(u, 0, None) * (partes > 0)


def altura_inflada(mascara: np.ndarray, gris: np.ndarray | None, inflado: float,
                   detalle: float, perfil: str = "borde", partes: np.ndarray | None = None,
                   mezcla_piezas: float = 0.5) -> tuple[np.ndarray, float]:
    """Mapa de alturas (en píxeles) + relieve fino de la foto. Devuelve (altura float32, altura máxima).

    perfil="borde": con d = distancia al borde, h = R * sqrt(1 - (1 - d/R)^2) para d < R (un cuarto de
    círculo) y meseta a partir de ahí. Sirve para marcos y para figuras con interior casi plano.

    perfil="piezas": como "globo", pero además cada pieza de color se abomba por separado (cojines con surcos
    entre ellos), como las partes de un juguete.

    perfil="globo": inflado de Poisson, h = sqrt(u) con laplaciano(u) = -1. La curvatura es continua en
    todo el cuerpo, como un globo o un muñeco de peluche, y la luz recorre toda la figura.
    """
    # El borde de la imagen cuenta como "afuera": así una máscara que ocupa todo el cuadro no da distancias infinitas.
    relleno = np.pad((mascara > 0).astype(np.uint8), 1)
    dist = cv2.distanceTransform(relleno, cv2.DIST_L2, 5)[1:-1, 1:-1]
    dmax = float(dist.max())
    if perfil in ("globo", "piezas"):
        u = np.clip(_poisson_inflado(mascara), 0, None)
        forma = np.sqrt(u)
        forma = forma / max(float(forma.max()), 1e-6)
        if perfil == "piezas" and partes is not None:
            q = np.sqrt(_poisson_partes(partes))
            q = q / max(float(q.max()), 1e-6)
            forma = (1.0 - mezcla_piezas) * forma + mezcla_piezas * q     # cuerpo redondeado + cada pieza abombada
        radio = max(2.0, inflado * dmax * 1.3)
        altura = forma * radio * (mascara > 0)
    else:
        radio = max(2.0, inflado * dmax)
        t = np.clip(dist / radio, 0.0, 1.0)
        altura = np.sqrt(1.0 - (1.0 - t) ** 2) * radio

    if detalle > 0 and gris is not None:
        sigma_fino = max(1.0, min(gris.shape) / 400)
        fino = cv2.GaussianBlur(gris, (0, 0), sigma_fino)
        local = cv2.GaussianBlur(gris, (0, 0), max(6.0, radio * 0.5))
        altura = altura + detalle * radio * 0.25 * (fino - local) * (dist > 0)
    return altura.astype(np.float32), radio


def normales(altura: np.ndarray, relieve: float = 1.0) -> np.ndarray:
    """Normales unitarias (alto, ancho, 3) con componentes (x, y, z); z apunta al espectador."""
    suave = cv2.GaussianBlur(altura, (0, 0), 1.2)
    gx = cv2.Sobel(suave, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(suave, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    n = np.dstack([-gx * relieve, -gy * relieve, np.ones_like(gx)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)
