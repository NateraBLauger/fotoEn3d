"""UN SOLO COMANDO: cada elemento de la imagen sobresale del fondo con efecto 3D (sin agregar nada nuevo):

    python demo.py imagen.jpg          # guarda imagen_3d.png

Opcional (solo si querés cambiar algo):
    python demo.py foto.jpg
    python demo.py dibujo.png --preset juguete          # más volumen y brillo
    python demo.py dibujo.png --preset arcilla          # look de juguete 3D: sin contornos, piezas de plástico
    python demo.py dibujo.png --preset diorama          # lo mismo, en una escena con pared, piso y nubes
    python demo.py foto.jpg --marco --fondo degradado --salida resultado_marco.png
    python demo.py ilustracion.png --objetos --sombra-piso 0.6
    python demo.py foto.jpg --luz-x 0.5 --brillo 0.6 --inflado 0.8
"""
import argparse
import os

import cv2

from foto3d import FONDOS, PERFILES, PRESETS, Foto3D, Parametros3D, elegir_preset


def main():
    ap = argparse.ArgumentParser(description="Convierte una foto en una imagen con efecto 3D")
    ap.add_argument("imagen")
    ap.add_argument("--salida", default=None, help="archivo de salida (por defecto <nombre>_3d.png)")
    ap.add_argument("--preset", choices=["auto"] + list(PRESETS), default="auto",
                    help="auto (por defecto) | elementos | elementos_foto | arcilla | juguete | suave | intenso | diorama")
    ap.add_argument("--marco", action="store_true", help="el sujeto sale de un marco de madera")
    ap.add_argument("--objetos", action="store_true", help="cada objeto con su volumen y sombras entre planos")
    ap.add_argument("--quitar-sombras", action="store_true", help="ignorar las sombras que ya trae la foto")
    ap.add_argument("--fondo", choices=FONDOS, default=None)
    ap.add_argument("--perfil", choices=PERFILES, default=None)
    for nombre, tipo, ayuda in [("inflado", float, "altura del volumen (0 a 1)"), ("relieve", float, "fuerza del relieve"),
                                ("brillo", float, "reflejo especular"), ("grosor", float, "espesor del borde"),
                                ("luz-x", float, "dirección de la luz en x"), ("luz-y", float, "dirección de la luz en y"),
                                ("sombra", float, "opacidad de la sombra (0 a 1)"),
                                ("sombra-piso", float, "sombra en el piso (0 a 1)"), ("nubes", int, "cantidad de nubes (0 a 6)"),
                                ("partes", int, "cantidad de colores para separar piezas (2 a 20)"),
                                ("quitar-contornos", float, "borrar contornos negros (0 a 1)"),
                                ("aplanar-color", float, "colores lisos (0 a 1)")]:
        ap.add_argument(f"--{nombre}", type=tipo, default=None, help=ayuda)
    ap.add_argument("--mascara", default=None, help="imagen en blanco y negro con el sujeto en blanco")
    ap.add_argument("--etapas", action="store_true", help="guardar también máscara, altura y normales")
    a = ap.parse_args()

    # Solo se pisan los valores del preset que se pasaron explícitamente.
    extra = dict(perfil=a.perfil, inflado=a.inflado, relieve=a.relieve, brillo=a.brillo, grosor=a.grosor,
                 luz_x=a.luz_x, luz_y=a.luz_y, sombra_opacidad=a.sombra, sombra_piso=a.sombra_piso,
                 nubes=a.nubes, fondo=a.fondo, partes=a.partes, quitar_contornos=a.quitar_contornos,
                 aplanar_color=a.aplanar_color)
    extra = {k: v for k, v in extra.items() if v is not None}
    if a.marco:
        extra["marco"] = True
    if a.objetos:
        extra["capas"] = "objetos"
    if a.quitar_sombras:
        extra["quitar_sombras"] = True
    if a.mascara:
        extra["mascara_sujeto"] = cv2.imread(a.mascara, cv2.IMREAD_GRAYSCALE)
    nombre = elegir_preset(a.imagen) if a.preset == "auto" else a.preset
    print("Preset:", nombre)
    r = Foto3D(Parametros3D.preset(nombre, **extra)).procesar(a.imagen)

    base = os.path.splitext(a.imagen)[0]
    salida = a.salida or f"{base}_3d.png"
    cv2.imwrite(salida, r.imagen)
    if a.etapas:
        cv2.imwrite(f"{base}_mascara.png", r.mascara)
        cv2.imwrite(f"{base}_altura.png", r.mapa_altura)
        cv2.imwrite(f"{base}_normales.png", r.mapa_normales)
        if r.mapa_partes is not None:
            cv2.imwrite(f"{base}_partes.png", r.mapa_partes)
    print("Guardado:", salida, "| tiempos (s):", r.tiempos)
    for aviso in r.advertencias:
        print("Aviso:", aviso)


if __name__ == "__main__":
    main()
