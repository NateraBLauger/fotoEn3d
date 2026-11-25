"""Clase principal: convierte una foto en una imagen con look de render 3D."""
from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass, field
from typing import Union

import cv2
import numpy as np

from .foto3d_backend.escena import construir_escena, construir_fondo, construir_marco, crear_nubes, pintar_marco
from .foto3d_backend.iluminacion import iluminar
from .foto3d_backend.parametros import Parametros3D
from .foto3d_backend.segmentacion import alfa_suave, mascara_auto, separar_objetos
from .foto3d_backend.estilo import (aplanar_color, colorear_partes, indice_de_textura, quitar_contornos,
                     segmentar_partes, separar_elementos)
from .foto3d_backend.sombras import extruir, sombra_piso, sombra_total
from .foto3d_backend.volumen import altura_inflada, normales

Imagen = Union[str, os.PathLike, np.ndarray]


@dataclass
class Resultado3D:
    """Salida de `Foto3D.procesar`. Las imágenes son BGR uint8 (convención OpenCV)."""

    imagen: np.ndarray                 # el resultado final
    sujeto: np.ndarray                 # solo el sujeto ya iluminado (sin fondo ni sombras)
    fondo: np.ndarray                  # fondo usado
    mascara: np.ndarray                # uint8 0/255
    alfa: np.ndarray                   # float32 [0, 1]
    mapa_altura: np.ndarray            # uint8 0-255, para visualizar el volumen
    mapa_normales: np.ndarray          # uint8 BGR, para visualizar la dirección de la superficie
    mapa_partes: np.ndarray | None = None   # uint8 BGR: las piezas de color (perfil "piezas")
    albedo: np.ndarray | None = None   # imagen de color usada (sin contornos y aplanada si corresponde)
    n_objetos: int = 1                 # objetos tratados como bloques independientes
    tiempos: dict[str, float] = field(default_factory=dict)
    advertencias: list[str] = field(default_factory=list)


class Foto3D:
    """Transforma una foto en una "foto 3D": volumen, luz, brillo, grosor, sombra y marco opcional.

    No es una reconstrucción 3D real: es una aproximación 2.5D (inflado de la silueta + iluminación).

        resultado = Foto3D().procesar("foto.jpg")
        cv2.imwrite("salida.png", resultado.imagen)
    """

    def __init__(self, parametros: Parametros3D | None = None):
        self.parametros = parametros or Parametros3D()

    @staticmethod
    def _preparar(imagen: Imagen, lado_max: int) -> np.ndarray:
        if isinstance(imagen, (str, os.PathLike)):
            img = cv2.imread(os.fspath(imagen), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"No se pudo leer la imagen: {imagen}")
        elif isinstance(imagen, np.ndarray):
            img = imagen
            if img.dtype != np.uint8:
                img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif not (img.ndim == 3 and img.shape[2] == 3):
                raise ValueError("La imagen debe ser gris, BGR o BGRA.")
        else:
            raise TypeError("imagen debe ser una ruta de archivo o un arreglo de NumPy (BGR).")
        alto, ancho = img.shape[:2]
        if max(alto, ancho) > lado_max:
            e = lado_max / max(alto, ancho)
            img = cv2.resize(img, (round(ancho * e), round(alto * e)), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    def procesar(self, imagen: Imagen, parametros: Parametros3D | None = None) -> Resultado3D:
        p = parametros or self.parametros
        p.validar()
        tiempos: dict[str, float] = {}

        def medir(nombre, t0):
            tiempos[nombre] = round(time.perf_counter() - t0, 3)

        # 1) Sujeto
        t0 = time.perf_counter()
        img = self._preparar(imagen, p.lado_max)
        lado = max(img.shape[:2])
        advertencias: list[str] = []
        if p.mascara_sujeto is not None:
            m = p.mascara_sujeto
            if m.ndim == 3:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mascara = np.where(m > 127, 255, 0).astype(np.uint8)
        else:
            mascara, advertencias = mascara_auto(img, quitar_sombras=p.quitar_sombras)
        if not mascara.any():
            raise ValueError("La máscara del sujeto está vacía.")

        bbox = cv2.boundingRect(mascara)
        grosor_marco = max(3, int(round(p.grosor_marco * lado)))
        escena = p.fondo == "escena"
        arriba = abajo = izq = der = 0
        if p.marco:   # se agranda el lienzo para que entre el marco completo
            arriba = abajo = izq = der = int(np.ceil(abs(p.margen_marco) * bbox[2] + grosor_marco + 8))
        if escena:    # más aire arriba (nubes) que a los costados y abajo, para que el objeto ocupe el cuadro
            g = max(bbox[2], bbox[3])
            arriba, abajo = max(arriba, int(0.26 * g)), max(abajo, int(0.10 * g))
            izq, der = max(izq, int(0.14 * g)), max(der, int(0.14 * g))
        if arriba or abajo or izq or der:
            img = cv2.copyMakeBorder(img, arriba, abajo, izq, der, cv2.BORDER_REPLICATE)
            mascara = cv2.copyMakeBorder(mascara, arriba, abajo, izq, der, cv2.BORDER_CONSTANT, value=0)
            bbox = (bbox[0] + izq, bbox[1] + arriba, bbox[2], bbox[3])
        alto, ancho = img.shape[:2]
        alfa = alfa_suave(img, mascara)
        # Estilo: sin contornos y con colores lisos. El brillo se calcula sobre este color.
        albedo = aplanar_color(quitar_contornos(img, mascara, p.quitar_contornos), p.aplanar_color)
        necesita_partes = p.perfil == "piezas" or p.capas == "elementos"
        partes_img = segmentar_partes(albedo, mascara, p.partes) if necesita_partes else None
        if necesita_partes and indice_de_textura(img, mascara) > 0.45:
            advertencias.append("La imagen tiene mucha textura o detalle fino. El estilo de piezas (presets elementos, "
                                "arcilla y diorama) funciona mejor con dibujos de colores lisos; para fotos probá el "
                                "preset \"elementos_foto\" o \"juguete\".")
        if p.capas == "elementos":
            objetos = separar_elementos(partes_img, p.max_objetos)
        elif p.capas == "objetos":
            objetos = separar_objetos(mascara, img, p.separar_tocados, p.sensibilidad_separacion, p.max_objetos)
        else:
            objetos = [mascara]
        medir("sujeto", t0)

        # 2) Escena base: fondo y marco
        t0 = time.perf_counter()
        modo = p.fondo if p.fondo != "auto" else ("degradado" if p.marco else "original")
        if escena:
            fondo = construir_escena((alto, ancho), bbox, p)
        else:
            fondo = construir_fondo(img, mascara, modo, p.color_fondo)
        nubes = crear_nubes((alto, ancho), bbox, mascara, p.nubes, p) if escena and p.nubes > 0 else []
        lienzo = fondo.copy()

        direccion = np.array([-p.luz_x, -p.luz_y], np.float32)
        direccion = direccion / max(float(np.linalg.norm(direccion)), 1e-6)
        dx, dy = direccion * p.sombra_desplazamiento * lado
        sigma = p.sombra_desenfoque * lado

        def sombrear(base, sombra):
            return base * (1.0 - 0.9 * sombra[..., None])

        marco = None
        corte = np.ones((alto, 1), np.float32)
        if p.marco:
            marco = construir_marco((alto, ancho), bbox, grosor_marco, p)
            v = marco["ventana"][..., None]
            lienzo = lienzo * (1 - v) + (lienzo * 0.6 + 0.4) * v          # la ventana se ve más clara
            a_marco = (marco["marco"] > 0).astype(np.float32)
            lienzo = sombrear(lienzo, sombra_total(a_marco, dx * 0.6, dy * 0.6, sigma * 0.7,
                                                   p.sombra_opacidad, p.sombra_contacto * 0.5))
            color_marco = pintar_marco(marco["marco"], p)
            lienzo = lienzo * (1 - a_marco[..., None]) + color_marco * a_marco[..., None]
            # El sujeto sale por la ventana: lo que está por debajo del borde inferior queda oculto.
            corte = (np.arange(alto)[:, None] < marco["y_corte"]).astype(np.float32)
        medir("escena_base", t0)

        # 3) Cada objeto, del más lejano al más cercano: volumen, luz, sombras y grosor.
        t0 = time.perf_counter()
        gris = cv2.cvtColor(albedo, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        altura_total = np.zeros((alto, ancho), np.float32)
        normal_total = np.zeros((alto, ancho, 3), np.float32)
        normal_total[..., 2] = 1.0
        sujeto_vis = albedo.astype(np.float32) / 255.0
        pasos = int(round(p.grosor * lado))
        inclinacion = float(-p.luz_x / p.luz_z * 0.3)
        nucleo = np.ones((3, 3), np.uint8)

        blanco = np.full_like(img, 250)
        # Las nubes se ven blancas y suaves: menos contraste y brillo que el sujeto.
        p_nube = dataclasses.replace(p, ambiente=max(p.ambiente, 0.62), brillo=0.15, relieve=1.1, oclusion=0.0, luz_borde=0.1)
        trabajo = [(m, True) for m in nubes] + [(o, False) for o in objetos]       # las nubes van al fondo
        for obj, es_nube in trabajo:
            if es_nube:
                alfa_i = cv2.GaussianBlur(obj.astype(np.float32) / 255.0, (0, 0), 1.0)
            elif len(objetos) == 1:
                alfa_i = alfa
            else:
                pesos = cv2.GaussianBlur(cv2.dilate(obj, nucleo).astype(np.float32) / 255.0, (0, 0), 1.0)
                alfa_i = np.clip(alfa * pesos, 0, 1)
            alfa_i = alfa_i * corte

            partes_obj = None if (es_nube or partes_img is None) else partes_img * (obj > 0)
            altura, _ = altura_inflada(obj, None if es_nube else gris, p.inflado, 0.0 if es_nube else p.detalle,
                                       "globo" if (es_nube and p.perfil == "piezas") else p.perfil, partes_obj, p.mezcla_piezas)
            norm = normales(altura, p.relieve)
            color = iluminar(blanco if es_nube else albedo, norm, altura, alfa_i, p_nube if es_nube else p)

            base_y = float(np.where(obj.any(axis=1))[0].max())
            if p.sombra_piso > 0 and base_y > 0.55 * alto:        # solo objetos apoyados en la parte baja
                lienzo = sombrear(lienzo, sombra_piso(alfa_i, base_y - 0.015 * lado, inclinacion, 0.22, sigma, p.sombra_piso * 0.7))
            # La sombra cae sobre el fondo y sobre los objetos más lejanos (ya dibujados).
            lienzo = sombrear(lienzo, sombra_total(alfa_i, dx, dy, sigma, p.sombra_opacidad, p.sombra_contacto))

            if pasos > 0:
                ex, ey = direccion * pasos
                pared_c, pared_a = extruir(color, alfa_i, ex, ey, pasos, p.oscurecer_grosor)
                pared_a = pared_a * corte
                lienzo = lienzo * (1 - pared_a[..., None]) + pared_c * pared_a[..., None]
            lienzo = lienzo * (1 - alfa_i[..., None]) + color * alfa_i[..., None]

            sel = alfa_i > 0.5
            altura_total[sel], normal_total[sel] = altura[sel], norm[sel]
            sujeto_vis = sujeto_vis * (1 - alfa_i[..., None]) + color * alfa_i[..., None]

        if marco is not None:                                            # barra inferior por delante
            b = (marco["barra"] > 0).astype(np.float32)[..., None]
            lienzo = lienzo * (1 - b) + color_marco * b
        medir("objetos", t0)

        a8 = lambda x: np.clip(x * 255, 0, 255).astype(np.uint8)
        tiempos["total"] = round(sum(tiempos.values()), 3)
        return Resultado3D(
            imagen=a8(lienzo), sujeto=a8(sujeto_vis), fondo=a8(fondo), mascara=mascara, alfa=alfa,
            mapa_altura=a8(altura_total / max(float(altura_total.max()), 1e-6)),
            mapa_normales=a8(normal_total[..., ::-1] * 0.5 + 0.5),
            mapa_partes=None if partes_img is None else colorear_partes(partes_img), albedo=albedo,
            n_objetos=len(objetos) + len(nubes), tiempos=tiempos, advertencias=advertencias)


def elegir_preset(imagen: Imagen, lado_max: int = 400) -> str:
    """Elige el preset de diorama que conviene a la imagen.

    "elementos" para dibujos de colores lisos (cada pieza de color sobresale del fondo) y "elementos_foto" para
    fotos y renders con mucho detalle, donde separar en piezas se ve "derretido". Ninguno agrega elementos nuevos.
    """
    img = Foto3D._preparar(imagen, lado_max)
    mascara, _ = mascara_auto(img)
    return "elementos_foto" if indice_de_textura(img, mascara) > 0.45 else "elementos"
