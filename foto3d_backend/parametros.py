"""Parámetros configurables del módulo Foto3D."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FONDOS = ("auto", "original", "desenfocado", "degradado", "color", "escena")
MODOS_CAPAS = ("una", "objetos", "elementos")
PERFILES = ("borde", "globo", "piezas")

# Combinaciones listas para usar. "suave" es el valor por defecto; los otros marcan más el volumen.
PRESETS = {
    "suave": dict(perfil="globo", inflado=0.8, relieve=1.5, ambiente=0.4, brillo=0.5, dureza_brillo=40.0),
    "juguete": dict(perfil="globo", inflado=0.9, relieve=2.0, ambiente=0.3, brillo=0.7, dureza_brillo=25.0),
    # Look de juguete 3D: sin contornos, colores lisos y cada pieza de color abombada por separado.
    "arcilla": dict(perfil="piezas", quitar_contornos=1.0, aplanar_color=1.0, partes=7, inflado=0.95, relieve=2.2,
                    ambiente=0.42, brillo=0.8, dureza_brillo=40.0, brillo_suave=0.35, saturacion=1.15,
                    grosor=0.012, sombra_opacidad=0.45, luz_borde=0.3, oclusion=0.25),
    # Cada elemento de la imagen (cada pieza de color) sobresale del fondo con volumen, grosor y sombra propios.
    # No se agrega nada a la imagen: el fondo es el original.
    "elementos": dict(capas="elementos", perfil="globo", quitar_contornos=1.0, aplanar_color=1.0, partes=6,
                      max_objetos=12, inflado=0.9, relieve=2.0, ambiente=0.42, brillo=0.8, dureza_brillo=40.0,
                      brillo_suave=0.35, saturacion=1.15, luz_borde=0.3, oclusion=0.25, grosor=0.02,
                      oscurecer_grosor=0.65, sombra_opacidad=0.7, sombra_desplazamiento=0.05,
                      sombra_desenfoque=0.035, sombra_contacto=0.45),
    # Para fotos o renders con mucho detalle: cada objeto separado del fondo, sin separar en piezas de color.
    "elementos_foto": dict(capas="objetos", perfil="globo", inflado=0.9, relieve=2.0, ambiente=0.35, brillo=0.7,
                           dureza_brillo=30.0, grosor=0.02, oscurecer_grosor=0.65, sombra_opacidad=0.7,
                           sombra_desplazamiento=0.05, sombra_desenfoque=0.035, sombra_contacto=0.45),
    "diorama": dict(perfil="piezas", quitar_contornos=1.0, aplanar_color=1.0, partes=7, inflado=0.95, relieve=2.2,
                    ambiente=0.42, brillo=0.8, dureza_brillo=40.0, brillo_suave=0.35, saturacion=1.15,
                    luz_borde=0.3, oclusion=0.25, grosor=0.02, oscurecer_grosor=0.65,
                    sombra_opacidad=0.5, sombra_desplazamiento=0.045, sombra_desenfoque=0.03,
                    fondo="escena", nubes=3, sombra_piso=0.7),
    # Igual que "diorama" pero para fotos y renders con mucho detalle (sin separar en piezas).
    "diorama_foto": dict(perfil="globo", inflado=0.9, relieve=2.0, ambiente=0.3, brillo=0.7, dureza_brillo=25.0,
                         grosor=0.02, oscurecer_grosor=0.65, sombra_opacidad=0.5, sombra_desplazamiento=0.045,
                         sombra_desenfoque=0.03, fondo="escena", nubes=3, sombra_piso=0.7),
    "intenso": dict(perfil="globo", inflado=1.0, relieve=2.5, ambiente=0.2, brillo=0.9, dureza_brillo=20.0,
                    luz_borde=0.4, oclusion=0.5),
}


@dataclass
class Parametros3D:
    """Todo lo que la aplicación puede ajustar. Los valores por defecto dan un look de juguete 3D."""

    lado_max: int = 900                      # tamaño de trabajo (lado mayor, en px)

    # --- Sujeto ---
    mascara_sujeto: Optional[np.ndarray] = None   # máscara propia (0 / 255); si no, se detecta sola

    quitar_sombras: bool = False             # trata las sombras del fondo como fondo (falla con objetos grises sobre blanco)

    # --- Capas ---
    capas: str = "una"                       # una = todo el sujeto es un bloque | objetos = cada objeto con su volumen y sus sombras
    separar_tocados: bool = True             # en modo objetos, divide objetos que se tocan (watershed)
    max_objetos: int = 9                     # en modo objetos, cantidad máxima de objetos independientes (el resto se agrupa)
    sensibilidad_separacion: float = 0.6     # menor = separa más (0.4 separa piezas de distinto tamaño; arriesga cortar personajes)

    # --- Volumen ---
    perfil: str = "globo"                    # globo = curvatura continua en todo el cuerpo | borde = solo el contorno redondeado
    inflado: float = 0.8                     # 0 = casi plano, 1 = muy redondeado
    partes: int = 8                          # perfil "piezas": cuántos colores usar para separar las piezas
    mezcla_piezas: float = 0.5               # perfil "piezas": 0 = solo el cuerpo, 1 = solo las piezas
    quitar_contornos: float = 0.0            # 0..1: borra los contornos negros del dibujo
    aplanar_color: float = 0.0               # 0..1: colores lisos por zonas (quita texturas y degradados finos)
    relieve: float = 1.5                     # fuerza del relieve en la luz
    detalle: float = 0.25                    # cuánto marca la foto el relieve fino (ojos, pliegues)

    # --- Luz ---
    luz_x: float = -0.5                      # dirección de la luz (x derecha, y abajo, z hacia el espectador)
    luz_y: float = -0.6
    luz_z: float = 0.8
    ambiente: float = 0.4                    # luz base (más alto = menos contraste)
    brillo: float = 0.5                      # reflejo especular (aspecto plástico/arcilla brillante)
    dureza_brillo: float = 40.0              # más alto = reflejo más chico y puntual
    luz_borde: float = 0.25                  # luz de contorno (separa al sujeto del fondo)
    brillo_suave: float = 0.25               # reflejo ancho y tenue, como plástico pulido (0 = apagado)
    oclusion: float = 0.35                   # oscurece las cavidades
    saturacion: float = 1.1

    # --- Grosor (borde tipo figura recortada) ---
    grosor: float = 0.012                    # fracción del lado mayor; 0 = sin grosor
    oscurecer_grosor: float = 0.6            # 1 = sin oscurecer

    # --- Sombras ---
    sombra_opacidad: float = 0.45
    sombra_desplazamiento: float = 0.03      # fracción del lado mayor
    sombra_desenfoque: float = 0.02          # fracción del lado mayor
    sombra_contacto: float = 0.35
    sombra_piso: float = 0.0                 # sombra proyectada en el piso para objetos apoyados en la parte baja de la foto (0 = apagada)

    # --- Fondo ---
    fondo: str = "auto"                      # auto | original | desenfocado | degradado | color
    color_fondo: tuple = (235, 232, 228)     # BGR, solo para fondo="color"

    # --- Escena de diorama (fondo="escena"): pared + piso + nubes con volumen ---
    color_pared: tuple = (214, 212, 143)     # BGR (celeste)
    color_piso: tuple = (107, 107, 255)      # BGR (coral)
    horizonte: float = 0.55                  # dónde empieza el piso, como fracción de la altura del sujeto
    inclinacion_horizonte: float = 0.12      # pendiente de la línea del piso (0 = horizontal)
    nubes: int = 0                           # cantidad de nubes decorativas (0 a 6)

    # --- Marco del que sale el sujeto ---
    marco: bool = False
    grosor_marco: float = 0.04               # fracción del lado mayor
    color_marco: tuple = (55, 75, 110)       # BGR (marrón madera)
    sobresale: float = 0.22                  # cuánto de la parte alta del sujeto queda fuera del marco
    bajo_marco: float = 0.05                 # parte baja del sujeto tapada por la barra inferior
    margen_marco: float = 0.06               # espacio entre sujeto y marco, como fracción del ancho del sujeto

    @classmethod
    def preset(cls, nombre: str, **cambios) -> "Parametros3D":
        """Parámetros de un preset ("suave", "juguete" o "intenso"); `cambios` pisa cualquier valor."""
        if nombre not in PRESETS:
            raise ValueError(f"preset debe ser uno de {tuple(PRESETS)}.")
        return cls(**{**PRESETS[nombre], **cambios})

    def validar(self) -> None:
        if self.lado_max < 64:
            raise ValueError("lado_max debe ser al menos 64 px.")
        if self.perfil not in PERFILES:
            raise ValueError(f"perfil debe ser uno de {PERFILES}.")
        if not 2 <= self.partes <= 20:
            raise ValueError("partes debe estar entre 2 y 20.")
        for n in ("mezcla_piezas", "quitar_contornos", "aplanar_color"):
            if not 0 <= getattr(self, n) <= 1:
                raise ValueError(f"{n} debe estar entre 0 y 1.")
        if self.brillo_suave < 0:
            raise ValueError("brillo_suave no puede ser negativo.")
        if not 0 <= self.inflado <= 1:
            raise ValueError("inflado debe estar entre 0 y 1.")
        if self.relieve < 0 or self.detalle < 0:
            raise ValueError("relieve y detalle no pueden ser negativos.")
        if self.luz_z <= 0:
            raise ValueError("luz_z debe ser mayor que 0 (la luz debe venir hacia el espectador).")
        if abs(self.luz_x) + abs(self.luz_y) + abs(self.luz_z) == 0:
            raise ValueError("La dirección de la luz no puede ser nula.")
        if not 0 <= self.ambiente < 1:
            raise ValueError("ambiente debe estar en [0, 1).")
        for n in ("brillo", "luz_borde", "oclusion", "saturacion", "sombra_contacto"):
            if getattr(self, n) < 0:
                raise ValueError(f"{n} no puede ser negativo.")
        if self.dureza_brillo < 1:
            raise ValueError("dureza_brillo debe ser al menos 1.")
        if not 0 <= self.grosor <= 0.1:
            raise ValueError("grosor debe estar entre 0 y 0.1.")
        if not 0 < self.oscurecer_grosor <= 1:
            raise ValueError("oscurecer_grosor debe estar en (0, 1].")
        if not 0 <= self.sombra_opacidad <= 1:
            raise ValueError("sombra_opacidad debe estar entre 0 y 1.")
        if not 0 <= self.sombra_desplazamiento <= 0.2 or not 0 <= self.sombra_desenfoque <= 0.2:
            raise ValueError("sombra_desplazamiento y sombra_desenfoque deben estar entre 0 y 0.2.")
        if self.capas not in MODOS_CAPAS:
            raise ValueError(f"capas debe ser una de {MODOS_CAPAS}.")
        if not 1 <= self.max_objetos <= 40:
            raise ValueError("max_objetos debe estar entre 1 y 40.")
        if not 0.2 <= self.sensibilidad_separacion <= 0.9:
            raise ValueError("sensibilidad_separacion debe estar entre 0.2 y 0.9.")
        if not 0 <= self.sombra_piso <= 1:
            raise ValueError("sombra_piso debe estar entre 0 y 1.")
        if self.fondo not in FONDOS:
            raise ValueError(f"fondo debe ser uno de {FONDOS}.")
        if len(self.color_pared) != 3 or len(self.color_piso) != 3:
            raise ValueError("color_pared y color_piso deben ser tuplas BGR de 3 valores.")
        if not 0.2 <= self.horizonte <= 0.95:
            raise ValueError("horizonte debe estar entre 0.2 y 0.95.")
        if not -0.4 <= self.inclinacion_horizonte <= 0.4:
            raise ValueError("inclinacion_horizonte debe estar entre -0.4 y 0.4.")
        if not 0 <= self.nubes <= 6:
            raise ValueError("nubes debe estar entre 0 y 6.")
        if len(self.color_fondo) != 3 or len(self.color_marco) != 3:
            raise ValueError("Los colores deben ser tuplas BGR de 3 valores.")
        if not 0.01 <= self.grosor_marco <= 0.15:
            raise ValueError("grosor_marco debe estar entre 0.01 y 0.15.")
        if not 0 <= self.sobresale <= 0.6 or not 0 <= self.bajo_marco <= 0.3:
            raise ValueError("sobresale debe estar en [0, 0.6] y bajo_marco en [0, 0.3].")
        if not -0.2 <= self.margen_marco <= 0.5:
            raise ValueError("margen_marco debe estar entre -0.2 y 0.5.")
