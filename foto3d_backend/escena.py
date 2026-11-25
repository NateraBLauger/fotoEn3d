"""Etapa 5: escena. Fondo (sin el sujeto) y marco de madera del que puede salir el sujeto."""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from .iluminacion import iluminar
from .parametros import Parametros3D
from .volumen import altura_inflada, normales


# ----------------------------------------------------------------- fondo ---
def _rellenar_sujeto(img_bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Quita al sujeto reconstruyendo el fondo detrás (inpainting en baja resolución)."""
    alto, ancho = img_bgr.shape[:2]
    esc = min(1.0, 320 / max(alto, ancho))
    peq = img_bgr if esc == 1.0 else cv2.resize(img_bgr, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mascara, (peq.shape[1], peq.shape[0]), interpolation=cv2.INTER_NEAREST)
    k = 2 * max(3, int(round(0.012 * max(peq.shape[:2])))) + 1        # el halo propio del objeto también se quita
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    relleno = cv2.inpaint(peq, m, 5, cv2.INPAINT_TELEA)
    relleno = cv2.resize(relleno, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    kz = 2 * max(4, int(round(0.012 * max(alto, ancho)))) + 1
    zona = cv2.dilate(mascara, np.ones((kz, kz), np.uint8)).astype(np.float32) / 255.0
    zona = cv2.GaussianBlur(zona, (0, 0), kz / 4)[..., None]
    return (img_bgr.astype(np.float32) * (1 - zona) + relleno.astype(np.float32) * zona) / 255.0


def construir_fondo(img_bgr: np.ndarray, mascara: np.ndarray, modo: str, color_bgr: tuple) -> np.ndarray:
    """Fondo float32 [0, 1] BGR del tamaño de la imagen."""
    alto, ancho = img_bgr.shape[:2]
    lado = max(alto, ancho)
    if modo == "color":
        return np.ones((alto, ancho, 3), np.float32) * (np.array(color_bgr, np.float32) / 255.0)

    if modo == "degradado":
        franja = max(2, alto // 20)
        arriba = np.median(img_bgr[:franja].reshape(-1, 3), axis=0).astype(np.float32) / 255.0
        abajo = np.median(img_bgr[-franja:].reshape(-1, 3), axis=0).astype(np.float32) / 255.0
        t = np.linspace(0, 1, alto, dtype=np.float32)[:, None, None]
        fondo = arriba * (1 - t) + abajo * t
        fondo = np.repeat(fondo, ancho, axis=1)
        yy, xx = np.mgrid[0:alto, 0:ancho].astype(np.float32)
        r = np.sqrt(((xx - ancho / 2) / (ancho / 2)) ** 2 + ((yy - alto / 2) / (alto / 2)) ** 2)
        return np.clip(fondo * (1.06 - 0.14 * np.clip(r / 1.4, 0, 1))[..., None], 0, 1)

    sin = _rellenar_sujeto(img_bgr, mascara)
    if modo == "desenfocado":
        return cv2.GaussianBlur(sin, (0, 0), 0.02 * lado)
    return sin


# ----------------------------------------------------------------- marco ---
def construir_marco(forma: tuple[int, int], bbox: tuple[int, int, int, int], grosor_px: int,
                    p: Parametros3D) -> dict:
    """Geometría del marco alrededor del sujeto. La barra inferior se dibuja delante del sujeto."""
    alto, ancho = forma
    bx, by, bw, bh = bbox
    mx = p.margen_marco * bw
    x0, x1 = int(round(bx - mx)), int(round(bx + bw + mx))
    y0, y1 = int(round(by + p.sobresale * bh)), int(round(by + bh - p.bajo_marco * bh))
    t = grosor_px

    marco = np.zeros((alto, ancho), np.uint8)
    cv2.rectangle(marco, (x0 - t, y0 - t), (x1 + t - 1, y1 + t - 1), 255, -1)
    cv2.rectangle(marco, (x0, y0), (x1 - 1, y1 - 1), 0, -1)

    barra = np.zeros((alto, ancho), np.uint8)
    cv2.rectangle(barra, (x0 - t, y1), (x1 + t - 1, y1 + t - 1), 255, -1)
    ventana = np.zeros((alto, ancho), np.float32)
    ventana[y0:y1, x0:x1] = 1.0
    return dict(marco=marco, barra=barra, ventana=ventana, y_corte=y1)


def pintar_marco(mascara_marco: np.ndarray, p: Parametros3D, semilla: int = 3) -> np.ndarray:
    """Marco de madera con veta y bisel iluminado. Devuelve color float32 [0, 1] BGR."""
    alto, ancho = mascara_marco.shape
    rng = np.random.default_rng(semilla)
    veta = cv2.GaussianBlur(rng.normal(0, 1, (alto, ancho)).astype(np.float32), (0, 0), sigmaX=50, sigmaY=1.0)
    veta /= max(float(veta.std()), 1e-6)
    base = np.ones((alto, ancho, 3), np.float32) * (np.array(p.color_marco, np.float32) / 255.0)
    base = np.clip(base * (1.0 + 0.07 * veta[..., None]), 0, 1)

    alfa = (mascara_marco > 0).astype(np.float32)
    altura, _ = altura_inflada(mascara_marco, None, 0.8, 0.0)
    norm = normales(altura, 1.2)
    p_madera = dataclasses.replace(p, brillo=0.15, luz_borde=0.1, saturacion=1.0, oclusion=0.0)
    return iluminar((base * 255).astype(np.uint8), norm, altura, alfa, p_madera, con_oclusion=False)


# --------------------------------------------------------------- diorama ---
def construir_escena(forma: tuple[int, int], bbox: tuple[int, int, int, int], p: Parametros3D) -> np.ndarray:
    """Fondo de diorama: pared y piso de colores lisos separados por un horizonte inclinado."""
    alto, ancho = forma
    bx, by, bw, bh = bbox
    yy, xx = np.mgrid[0:alto, 0:ancho].astype(np.float32)
    y_h = by + p.horizonte * bh + p.inclinacion_horizonte * (xx - (bx + bw / 2))
    piso = cv2.GaussianBlur((yy >= y_h).astype(np.float32), (0, 0), 1.0)[..., None]
    pared = np.array(p.color_pared, np.float32) / 255.0
    suelo = np.array(p.color_piso, np.float32) / 255.0
    pared_img = pared * (1.04 - 0.10 * np.clip(yy / max(alto, 1), 0, 1))[..., None]
    piso_img = suelo * (1.03 - 0.10 * np.clip((yy - y_h) / (0.5 * alto), 0, 1))[..., None]
    return np.clip(pared_img * (1 - piso) + piso_img * piso, 0, 1)


def crear_nubes(forma: tuple[int, int], bbox: tuple[int, int, int, int], mascara: np.ndarray,
                n: int, p: Parametros3D, semilla: int = 11) -> list[np.ndarray]:
    """Máscaras de nubes (uint8 0/255) sobre la pared, sin tocar al sujeto ni caer en el piso."""
    alto, ancho = forma
    bx, by, bw, bh = bbox
    rng = np.random.default_rng(semilla)
    anclas = [(0.17, 0.17), (0.82, 0.22), (0.50, 0.07), (0.10, 0.36), (0.90, 0.40), (0.33, 0.28), (0.70, 0.10)]
    forma_nube = [(0.0, 0.0, 1.0), (-0.9, 0.35, 0.7), (0.95, 0.3, 0.75), (0.1, 0.55, 0.8)]
    cerca_sujeto = cv2.dilate(mascara, np.ones((9, 9), np.uint8))
    nubes: list[np.ndarray] = []
    for ax, ay in anclas:
        if len(nubes) >= n:
            break
        r0 = 0.085 * max(alto, ancho) * float(rng.uniform(0.85, 1.15))
        for factor in (1.0, 0.85, 0.7, 0.55):                # si no entra, se prueba una nube más chica
            r = r0 * factor
            cx = float(np.clip(ax * ancho, 2.2 * r, ancho - 2.2 * r))
            cy = float(np.clip(ay * alto, 1.6 * r, alto - 1.6 * r))
            m = np.zeros((alto, ancho), np.uint8)
            for dx, dy, k in forma_nube:
                cv2.circle(m, (int(cx + dx * r), int(cy + dy * r)), int(k * r), 255, -1)
            y_h = by + p.horizonte * bh + p.inclinacion_horizonte * (cx - (bx + bw / 2))
            base = int(np.where(m.any(axis=1))[0].max())
            solapa_nube = any((m & otra).any() for otra in nubes)
            if not (m & cerca_sujeto).any() and base <= y_h - 4 and not solapa_nube:
                nubes.append(m)
                break
    return nubes
