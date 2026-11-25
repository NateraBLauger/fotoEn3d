"""Etapa 3: iluminación (difusa + brillo especular + luz de borde + oclusión de cavidades)."""
from __future__ import annotations

import cv2
import numpy as np

from .parametros import Parametros3D


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def iluminar(color_bgr: np.ndarray, norm: np.ndarray, altura: np.ndarray, alfa: np.ndarray,
             p: Parametros3D, con_oclusion: bool = True) -> np.ndarray:
    """Devuelve el color iluminado (float32 [0, 1]). Fuera del alfa el color queda sin cambios."""
    img = color_bgr.astype(np.float32) / 255.0

    if p.saturacion != 1.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * p.saturacion, 0, 1)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    luz = _unit([p.luz_x, p.luz_y, p.luz_z])
    # Difusa: una superficie plana de frente queda con el color original (factor 1).
    difusa = p.ambiente + (1 - p.ambiente) * np.clip((norm * luz).sum(axis=2), 0, 1)
    plana = p.ambiente + (1 - p.ambiente) * float(luz[2])
    sombreado = (difusa / plana)[..., None]

    oclu = np.ones(alfa.shape, np.float32)
    if con_oclusion and p.oclusion > 0:
        cav = np.clip(cv2.GaussianBlur(altura, (0, 0), max(2.0, min(alfa.shape) / 60)) - altura, 0, None)
        escala = float(np.percentile(cav[alfa > 0.5], 99)) if (alfa > 0.5).any() else 0.0
        if escala > 1e-6:
            oclu = 1.0 - p.oclusion * np.clip(cav / escala, 0, 1)

    color = img * sombreado * oclu[..., None]

    # Brillo especular (Blinn-Phong); se resta el de una superficie plana para no lavar toda la imagen.
    mitad = _unit(luz + np.array([0, 0, 1], np.float32))
    nh = np.clip((norm * mitad).sum(axis=2), 0, 1)
    plano = float(mitad[2])
    espec = np.clip(nh ** p.dureza_brillo - plano ** p.dureza_brillo, 0, None) * p.brillo
    if p.brillo_suave > 0:                       # reflejo ancho y tenue: aspecto de plástico pulido
        amplio = max(2.0, p.dureza_brillo / 5)
        espec = espec + np.clip(nh ** amplio - plano ** amplio, 0, None) * p.brillo * p.brillo_suave

    # Luz de contorno desde atrás: realza el borde del lado opuesto a la luz principal.
    contra = _unit([-p.luz_x, -p.luz_y, 0.3])
    borde = np.clip((norm * contra).sum(axis=2), 0, 1) ** 2 * p.luz_borde

    luz_extra = (espec + borde)[..., None]
    iluminado = np.clip(color + luz_extra * (1.0 - color * 0.5), 0, 1)
    a = alfa[..., None]
    return iluminado * a + img * (1 - a)
