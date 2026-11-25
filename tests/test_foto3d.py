"""Pruebas del módulo. Ejecutar desde la carpeta del proyecto con:
    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foto3d import FONDOS, PRESETS, Foto3D, Parametros3D, elegir_preset
from foto3d.ejemplos import ilustracion_plana
from foto3d.escena import construir_escena, crear_nubes
from foto3d.estilo import aplanar_color, indice_de_textura, quitar_contornos, segmentar_partes, separar_elementos
from foto3d.segmentacion import alfa_suave, mascara_auto, separar_objetos
from foto3d.sombras import extruir, sombra_piso
from foto3d.volumen import _poisson_inflado, _poisson_partes, altura_inflada, normales


def imagen_prueba(ancho=320, alto=240):
    """Fondo con degradado suave y una elipse roja con un brillo, bien separada del fondo."""
    y = np.linspace(0, 1, alto, dtype=np.float32)[:, None, None]
    fondo = (np.array([200, 190, 170], np.float32) * (1 - y) + np.array([160, 170, 150], np.float32) * y)
    img = np.repeat(fondo, ancho, axis=1).astype(np.uint8)
    cv2.ellipse(img, (ancho // 2, alto // 2), (ancho // 5, alto // 3), 0, 0, 360, (40, 40, 200), -1)
    cv2.circle(img, (ancho // 2 - 15, alto // 2 - 25), 12, (90, 90, 240), -1)
    return img


def params(**kw):
    base = dict(lado_max=320)
    base.update(kw)
    return Parametros3D(**base)


class TestSegmentacion(unittest.TestCase):
    def test_detecta_el_sujeto(self):
        img = imagen_prueba()
        mascara, avisos = mascara_auto(img)
        self.assertEqual(mascara.shape, img.shape[:2])
        self.assertEqual(set(np.unique(mascara)), {0, 255})
        real = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(real, (160, 120), (64, 80), 0, 0, 360, 255, -1)
        interseccion = np.logical_and(mascara > 0, real > 0).sum()
        union = np.logical_or(mascara > 0, real > 0).sum()
        self.assertGreater(interseccion / union, 0.85)       # IoU
        self.assertEqual(avisos, [])

    def test_alfa_en_rango(self):
        img = imagen_prueba()
        mascara, _ = mascara_auto(img)
        alfa = alfa_suave(img, mascara)
        self.assertGreaterEqual(float(alfa.min()), 0.0)
        self.assertLessEqual(float(alfa.max()), 1.0)
        self.assertEqual(alfa.dtype, np.float32)


class TestVolumen(unittest.TestCase):
    def test_altura_mayor_en_el_centro_y_cero_fuera(self):
        m = np.zeros((100, 100), np.uint8)
        cv2.circle(m, (50, 50), 30, 255, -1)
        altura, radio = altura_inflada(m, None, 1.0, 0.0)
        self.assertGreater(altura[50, 50], altura[50, 25])
        self.assertEqual(float(altura[5, 5]), 0.0)
        self.assertGreater(radio, 2.0)

    def test_normales_unitarias_y_planas_fuera(self):
        m = np.zeros((100, 100), np.uint8)
        cv2.circle(m, (50, 50), 30, 255, -1)
        altura, _ = altura_inflada(m, None, 0.8, 0.0)
        n = normales(altura)
        np.testing.assert_allclose(np.linalg.norm(n, axis=2), 1.0, atol=1e-5)
        np.testing.assert_allclose(n[3, 3], [0, 0, 1], atol=1e-5)
        self.assertLess(n[50, 22, 2], 0.95)                  # el borde está inclinado

    def test_la_normal_del_borde_mira_hacia_afuera(self):
        m = np.zeros((100, 100), np.uint8)
        cv2.circle(m, (50, 50), 30, 255, -1)
        altura, _ = altura_inflada(m, None, 1.0, 0.0)
        n = normales(altura)
        self.assertLess(n[50, 24, 0], 0)                      # borde izquierdo: normal hacia la izquierda
        self.assertGreater(n[50, 76, 0], 0)                   # borde derecho: hacia la derecha


class TestSombras(unittest.TestCase):
    def test_extrusion_asoma_fuera_del_sujeto(self):
        alfa = np.zeros((80, 80), np.float32)
        alfa[30:50, 30:50] = 1
        color = np.ones((80, 80, 3), np.float32) * 0.8
        pared_c, pared_a = extruir(color, alfa, 6, 6, 6, 0.5)
        self.assertGreater(pared_a[55, 55], 0.9)              # fuera del sujeto, hacia abajo-derecha
        self.assertEqual(float(pared_a[10, 10]), 0.0)
        self.assertAlmostEqual(float(pared_c[55, 55, 0]), 0.4, places=2)


class TestFoto3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = imagen_prueba()
        cls.res = Foto3D(params()).procesar(cls.img)

    def test_salida_basica(self):
        r = self.res
        self.assertEqual(r.imagen.shape, self.img.shape)
        self.assertEqual(r.imagen.dtype, np.uint8)
        self.assertEqual(r.mapa_normales.shape, self.img.shape)
        self.assertGreater(np.abs(r.imagen.astype(int) - self.img.astype(int)).mean(), 1.0)
        self.assertEqual(r.advertencias, [])
        self.assertIn("total", r.tiempos)

    def test_lejos_del_sujeto_el_fondo_casi_no_cambia(self):
        esquina = np.abs(self.res.imagen[:20, :20].astype(int) - self.img[:20, :20].astype(int)).mean()
        self.assertLess(esquina, 6.0)

    def test_la_sombra_cae_del_lado_opuesto_a_la_luz(self):
        # Luz desde arriba-izquierda: la zona abajo-derecha del sujeto queda más oscura que arriba-izquierda.
        abajo_der = self.res.imagen[175:190, 215:230].astype(int).mean()
        arriba_izq = self.res.imagen[175:190, 90:105].astype(int).mean()
        self.assertLess(abajo_der, arriba_izq + 1)

    def test_es_determinista(self):
        otra = Foto3D(params()).procesar(self.img)
        np.testing.assert_array_equal(self.res.imagen, otra.imagen)

    def test_no_modifica_la_entrada(self):
        copia = self.img.copy()
        Foto3D(params()).procesar(self.img)
        np.testing.assert_array_equal(copia, self.img)

    def test_acepta_ruta_gris_y_bgra(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "f.png")
            cv2.imwrite(ruta, self.img)
            self.assertEqual(Foto3D(params()).procesar(ruta).imagen.shape, self.img.shape)
        gris = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.assertEqual(Foto3D(params()).procesar(gris).imagen.shape, self.img.shape)
        bgra = cv2.cvtColor(self.img, cv2.COLOR_BGR2BGRA)
        self.assertEqual(Foto3D(params()).procesar(bgra).imagen.shape, self.img.shape)

    def test_reduce_a_lado_max(self):
        grande = imagen_prueba(640, 480)
        r = Foto3D(params(lado_max=200)).procesar(grande)
        self.assertEqual(max(r.imagen.shape[:2]), 200)

    def test_mascara_propia(self):
        m = np.zeros(self.img.shape[:2], np.uint8)
        cv2.circle(m, (80, 80), 30, 255, -1)
        r = Foto3D(params(mascara_sujeto=m)).procesar(self.img)
        self.assertEqual(int((r.mascara > 0).sum()), int((m > 0).sum()))

    def test_todos_los_fondos(self):
        for f in FONDOS:
            r = Foto3D(params(fondo=f)).procesar(self.img)
            if f == "escena":                       # la escena agranda el lienzo para que entren pared, piso y nubes
                self.assertGreater(r.imagen.shape[0], self.img.shape[0], f)
            else:
                self.assertEqual(r.imagen.shape, self.img.shape, f)

    def test_marco_agranda_el_lienzo_y_dibuja_madera(self):
        r = Foto3D(params(marco=True)).procesar(self.img)
        self.assertGreater(r.imagen.shape[0], self.img.shape[0])
        self.assertGreater(r.imagen.shape[1], self.img.shape[1])
        self.assertEqual(r.fondo.shape, r.imagen.shape)

    def test_sin_grosor_ni_sombra(self):
        r = Foto3D(params(grosor=0, sombra_opacidad=0, sombra_contacto=0)).procesar(self.img)
        self.assertEqual(r.imagen.shape, self.img.shape)

    def test_errores_claros(self):
        modulo = Foto3D()
        with self.assertRaises(FileNotFoundError):
            modulo.procesar("no_existe.jpg")
        with self.assertRaises(TypeError):
            modulo.procesar(123)
        with self.assertRaises(ValueError):                   # máscara vacía
            modulo.procesar(self.img, Parametros3D(mascara_sujeto=np.zeros(self.img.shape[:2], np.uint8)))
        for malo in (dict(inflado=2), dict(luz_z=0), dict(fondo="x"), dict(grosor=0.5),
                     dict(sombra_opacidad=3), dict(ambiente=1.0), dict(dureza_brillo=0),
                     dict(grosor_marco=0.5), dict(color_fondo=(1, 2)), dict(lado_max=10)):
            with self.assertRaises(ValueError, msg=str(malo)):
                modulo.procesar(self.img, Parametros3D(**malo))

    def test_imagen_lisa_no_falla(self):
        lisa = np.full((120, 160, 3), 128, np.uint8)
        r = Foto3D(params()).procesar(lisa)
        self.assertEqual(r.imagen.shape, lisa.shape)
        self.assertTrue(r.advertencias)


class TestObjetos(unittest.TestCase):
    def test_separa_componentes_y_ordena_de_lejos_a_cerca(self):
        m = np.zeros((200, 300), np.uint8)
        cv2.circle(m, (60, 140), 30, 255, -1)         # base en y=170 (más cerca)
        cv2.circle(m, (220, 60), 30, 255, -1)         # base en y=90 (más lejos)
        img = np.zeros((200, 300, 3), np.uint8)
        objetos = separar_objetos(m, img)
        self.assertEqual(len(objetos), 2)
        bases = [int(np.where(o.any(axis=1))[0].max()) for o in objetos]
        self.assertEqual(bases, sorted(bases))

    def test_divide_dos_objetos_que_se_tocan(self):
        m = np.zeros((200, 300), np.uint8)
        cv2.circle(m, (110, 100), 45, 255, -1)
        cv2.circle(m, (190, 100), 45, 255, -1)
        img = np.zeros((200, 300, 3), np.uint8)
        self.assertEqual(len(separar_objetos(m, img, separar_tocados=True)), 2)
        self.assertEqual(len(separar_objetos(m, img, separar_tocados=False)), 1)

    def test_un_objeto_redondo_no_se_parte(self):
        m = np.zeros((200, 200), np.uint8)
        cv2.circle(m, (100, 100), 60, 255, -1)
        self.assertEqual(len(separar_objetos(m, np.zeros((200, 200, 3), np.uint8))), 1)

    def test_limita_la_cantidad_de_objetos(self):
        m = np.zeros((300, 300), np.uint8)
        for i in range(12):
            cv2.circle(m, (25 + 24 * (i % 6), 60 + 100 * (i // 6)), 9, 255, -1)
        self.assertLessEqual(len(separar_objetos(m, np.zeros((300, 300, 3), np.uint8), max_objetos=5)), 5)

    def test_sombra_de_piso_solo_debajo_de_la_base(self):
        alfa = np.zeros((200, 200), np.float32)
        alfa[60:120, 80:120] = 1
        s = sombra_piso(alfa, 119, 0.3, 0.3, 4.0, 1.0)
        self.assertLess(float(s[:100].max()), 0.01)
        self.assertGreater(float(s[120:150].max()), 0.2)

    def test_modo_objetos_en_ilustracion(self):
        img = ilustracion_plana()                       # 720 x 600, objetos en coordenadas absolutas
        r = Foto3D(params(lado_max=720, capas="objetos", sombra_piso=0.6)).procesar(img)
        self.assertGreaterEqual(r.n_objetos, 3)
        self.assertEqual(r.imagen.shape, img.shape)
        una = Foto3D(params(lado_max=720)).procesar(img)
        self.assertEqual(una.n_objetos, 1)
        self.assertGreater(np.abs(r.imagen.astype(int) - una.imagen.astype(int)).mean(), 0.05)

    def test_quitar_sombras_funciona(self):
        r = Foto3D(params(quitar_sombras=True)).procesar(imagen_prueba())
        self.assertEqual(r.imagen.shape, imagen_prueba().shape)

    def test_parametros_nuevos_invalidos(self):
        m = Foto3D()
        for malo in (dict(capas="x"), dict(sombra_piso=2), dict(sensibilidad_separacion=0.05)):
            with self.assertRaises(ValueError, msg=str(malo)):
                m.procesar(imagen_prueba(), Parametros3D(**malo))


class TestGlobo(unittest.TestCase):
    def setUp(self):
        self.m = np.zeros((120, 120), np.uint8)
        cv2.circle(self.m, (60, 60), 45, 255, -1)

    def test_poisson_es_una_cupula(self):
        u = _poisson_inflado(self.m)
        self.assertEqual(u.shape, self.m.shape)
        self.assertGreater(u[60, 60], 0)
        self.assertEqual(float(u[2, 2]), 0.0)
        self.assertGreaterEqual(u[60, 60], u.max() * 0.95)         # el máximo está en el centro

    def test_globo_crece_de_forma_continua_hasta_el_centro(self):
        altura, _ = altura_inflada(self.m, None, 0.8, 0.0, perfil="globo")
        fila = altura[60, 60:106]                                  # del centro hacia el borde
        self.assertTrue(np.all(np.diff(fila) <= 1e-3))             # nunca sube al alejarse del centro
        self.assertGreater(fila[0] - fila[len(fila) // 2], 1.0)    # no hay meseta: ya baja a mitad de camino
        self.assertEqual(float(altura[5, 5]), 0.0)

    def test_el_perfil_borde_sigue_siendo_una_meseta(self):
        altura, _ = altura_inflada(self.m, None, 0.5, 0.0, perfil="borde")
        self.assertAlmostEqual(float(altura[60, 60]), float(altura[60, 70]), places=3)

    def test_presets(self):
        for nombre in PRESETS:
            r = Foto3D(Parametros3D.preset(nombre, lado_max=240)).procesar(imagen_prueba(240, 180))
            if PRESETS[nombre].get("fondo") == "escena":          # la escena agranda el lienzo
                self.assertGreater(r.imagen.shape[0], 180, nombre)
            else:
                self.assertEqual(r.imagen.shape, (180, 240, 3), nombre)
        self.assertEqual(Parametros3D.preset("juguete", brillo=0.1).brillo, 0.1)
        with self.assertRaises(ValueError):
            Parametros3D.preset("no_existe")

    def test_mascara_que_ocupa_todo_el_cuadro(self):
        todo = np.full((80, 120), 255, np.uint8)
        for perfil in ("borde", "globo"):
            altura, radio = altura_inflada(todo, None, 0.8, 0.0, perfil=perfil)
            self.assertTrue(np.isfinite(altura).all(), perfil)
            self.assertLess(radio, 200, perfil)
            self.assertGreater(float(altura[40, 60]), float(altura[0, 0]), perfil)

    def test_perfil_invalido(self):
        with self.assertRaises(ValueError):
            Foto3D().procesar(imagen_prueba(), Parametros3D(perfil="x"))

    def test_mas_intenso_marca_mas_el_volumen(self):
        img = imagen_prueba()
        ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cambio = {}
        for nombre in ("suave", "intenso"):
            r = Foto3D(Parametros3D.preset(nombre, lado_max=320)).procesar(img)
            suj = cv2.cvtColor(r.sujeto, cv2.COLOR_BGR2GRAY).astype(np.float32)
            interior = cv2.erode(r.mascara, np.ones((15, 15), np.uint8)) > 0
            cambio[nombre] = float(np.abs(suj - ref)[interior].mean())   # cuánto cambia la luz dentro del sujeto
        self.assertGreater(cambio["intenso"], cambio["suave"])


class TestEscena(unittest.TestCase):
    def setUp(self):
        self.p = Parametros3D(fondo="escena", nubes=3, lado_max=320)
        self.forma = (400, 500)
        self.bbox = (180, 120, 140, 200)

    def test_escena_tiene_pared_arriba_y_piso_abajo(self):
        esc = construir_escena(self.forma, self.bbox, self.p)
        self.assertEqual(esc.shape, (400, 500, 3))
        pared = np.array(self.p.color_pared, np.float32) / 255
        piso = np.array(self.p.color_piso, np.float32) / 255
        self.assertLess(float(np.abs(esc[5, 5] - pared).max()), 0.12)
        self.assertLess(float(np.abs(esc[395, 495] - piso).max()), 0.12)

    def test_las_nubes_no_tocan_al_sujeto_ni_caen_en_el_piso(self):
        mascara = np.zeros(self.forma, np.uint8)
        x, y, w, h = self.bbox
        mascara[y:y + h, x:x + w] = 255
        nubes = crear_nubes(self.forma, self.bbox, mascara, 4, self.p)
        self.assertGreaterEqual(len(nubes), 1)
        self.assertLessEqual(len(nubes), 4)
        for n in nubes:
            self.assertFalse((n & mascara).any())
            self.assertLess(int(np.where(n.any(axis=1))[0].max()), y + self.p.horizonte * h)

    def test_nubes_en_cero_no_agrega_objetos(self):
        r = Foto3D(Parametros3D(fondo="escena", nubes=0, lado_max=320)).procesar(imagen_prueba())
        self.assertEqual(r.n_objetos, 1)

    def test_diorama_completo(self):
        r = Foto3D(Parametros3D.preset("diorama", lado_max=320)).procesar(imagen_prueba())
        self.assertGreater(r.n_objetos, 1)                                   # el sujeto y al menos una nube
        self.assertGreater(r.imagen.shape[0], 240)
        pared = np.array(r.fondo[3, 3], np.int32)
        self.assertLess(int(np.abs(pared - np.array(self.p.color_pared)).max()), 40)

    def test_parametros_de_escena_invalidos(self):
        m = Foto3D()
        for malo in (dict(nubes=9), dict(horizonte=0.0), dict(inclinacion_horizonte=2), dict(color_pared=(1, 2))):
            with self.assertRaises(ValueError, msg=str(malo)):
                m.procesar(imagen_prueba(), Parametros3D(**malo))


class TestEstiloArcilla(unittest.TestCase):
    def dibujo(self):
        """Disco rosa con contorno negro fino, una boca corta y una mancha clara."""
        img = np.full((240, 240, 3), 255, np.uint8)
        cv2.circle(img, (120, 120), 90, (170, 110, 230), -1)
        cv2.circle(img, (120, 120), 90, (20, 20, 20), 3)
        cv2.line(img, (105, 150), (135, 150), (20, 20, 20), 3)
        mascara = np.zeros((240, 240), np.uint8)
        cv2.circle(mascara, (120, 120), 92, 255, -1)
        return img, mascara

    def test_quita_el_contorno_largo_y_conserva_el_detalle_corto(self):
        img, mascara = self.dibujo()
        limpio = quitar_contornos(img, mascara, 1.0)
        anillo = np.zeros((240, 240), bool)
        cv2.circle(anillo.view(np.uint8), (120, 120), 90, 1, 3)
        gris = cv2.cvtColor(limpio, cv2.COLOR_BGR2GRAY).astype(int)
        orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(int)
        self.assertGreater(gris[anillo].mean(), orig[anillo].mean() + 40)      # el contorno se aclara
        self.assertLess(gris[150, 112:128].mean(), 90)                         # la boca sigue oscura

    def test_fuerza_cero_no_cambia_nada(self):
        img, mascara = self.dibujo()
        np.testing.assert_array_equal(quitar_contornos(img, mascara, 0.0), img)
        np.testing.assert_array_equal(aplanar_color(img, 0.0), img)

    def test_aplanar_reduce_el_ruido(self):
        rng = np.random.default_rng(0)
        img = np.clip(np.full((120, 120, 3), 150) + rng.normal(0, 12, (120, 120, 3)), 0, 255).astype(np.uint8)
        self.assertLess(float(aplanar_color(img, 1.0).std()), float(img.std()) * 0.6)

    def test_segmenta_bloques_de_color_en_piezas(self):
        img = np.full((200, 300, 3), 255, np.uint8)
        img[:, :100], img[:, 100:200], img[:, 200:] = (40, 40, 220), (60, 200, 60), (220, 80, 40)
        mascara = np.full((200, 300), 255, np.uint8)
        partes = segmentar_partes(img, mascara, k=3)
        self.assertEqual(len(np.unique(partes[partes > 0])), 3)
        self.assertNotEqual(int(partes[100, 50]), int(partes[100, 150]))

    def test_fusiona_tonos_parecidos_y_separa_colores_distintos(self):
        img = np.full((200, 300, 3), (200, 140, 220), np.uint8)             # rosa
        img[:, 100:200] = (196, 138, 218)                                    # otro rosa casi igual
        img[:, 200:] = (40, 40, 40)                                          # negro
        mascara = np.full((200, 300), 255, np.uint8)
        con = segmentar_partes(img, mascara, k=3, fusion_color=15.0)
        sin = segmentar_partes(img, mascara, k=3, fusion_color=0.0)
        self.assertEqual(int(con[100, 50]), int(con[100, 150]))               # los dos rosas son una pieza
        self.assertNotEqual(int(con[100, 50]), int(con[100, 250]))            # el negro queda aparte
        self.assertGreaterEqual(len(np.unique(sin[sin > 0])), len(np.unique(con[con > 0])))

    def test_separar_elementos_ordena_de_grande_a_chico(self):
        partes = np.zeros((100, 100), np.int32)
        partes[10:90, 10:60] = 1                                              # grande
        partes[10:30, 70:90] = 2                                              # chico
        partes[50:90, 70:95] = 3                                              # mediano
        els = separar_elementos(partes)
        areas = [int((e > 0).sum()) for e in els]
        self.assertEqual(areas, sorted(areas, reverse=True))
        self.assertEqual(len(els), 3)
        self.assertEqual(len(separar_elementos(partes, maximo=2)), 2)

    def test_modo_elementos_no_agrega_nada_nuevo(self):
        img, _ = self.dibujo()
        r = Foto3D(Parametros3D.preset("elementos", lado_max=240)).procesar(img)
        self.assertEqual(r.imagen.shape, img.shape)                           # no se agranda el lienzo
        self.assertGreaterEqual(r.n_objetos, 1)
        esquina = np.abs(r.imagen[:15, :15].astype(int) - img[:15, :15].astype(int)).mean()
        self.assertLess(esquina, 8.0)                                         # el fondo original se conserva

    def test_las_piezas_diminutas_se_unen_a_una_vecina(self):
        img = np.full((200, 200, 3), (60, 200, 60), np.uint8)
        img[95:100, 95:100] = (40, 40, 220)                                    # mota de 25 píxeles
        partes = segmentar_partes(img, np.full((200, 200), 255, np.uint8), k=2)
        cuentas = np.bincount(partes[partes > 0].ravel())[1:]
        self.assertGreaterEqual(int(cuentas.min()), 0.012 * 200 * 200 * 0.99)

    def test_poisson_por_piezas_deja_un_surco_en_el_borde_comun(self):
        partes = np.zeros((100, 200), np.int32)
        partes[10:90, 10:100], partes[10:90, 100:190] = 1, 2
        u = _poisson_partes(partes)
        self.assertEqual(float(u[50, 100]), 0.0)                              # borde entre las dos piezas
        self.assertGreater(float(u[50, 55]), 0.0)
        self.assertGreater(float(u[50, 145]), 0.0)
        self.assertEqual(float(u[5, 5]), 0.0)

    def test_perfil_piezas_marca_el_surco_entre_piezas(self):
        partes = np.zeros((100, 200), np.int32)
        partes[10:90, 10:100], partes[10:90, 100:190] = 1, 2
        mascara = (partes > 0).astype(np.uint8) * 255
        altura, _ = altura_inflada(mascara, None, 0.9, 0.0, perfil="piezas", partes=partes, mezcla_piezas=0.8)
        self.assertLess(float(altura[50, 100]), float(altura[50, 80]) * 0.7)
        con, _ = altura_inflada(mascara, None, 0.9, 0.0, perfil="globo")
        self.assertGreater(float(con[50, 100]), float(altura[50, 100]))        # sin piezas no hay surco

    def test_preset_arcilla_completo(self):
        img, _ = self.dibujo()
        r = Foto3D(Parametros3D.preset("arcilla", lado_max=240)).procesar(img)
        self.assertIsNotNone(r.mapa_partes)
        self.assertEqual(r.mapa_partes.shape, r.imagen.shape)
        self.assertIsNotNone(r.albedo)
        base = Foto3D(Parametros3D(lado_max=240)).procesar(img)
        self.assertGreater(np.abs(r.imagen.astype(int) - base.imagen.astype(int)).mean(), 1.0)

    def test_avisa_si_la_imagen_tiene_mucha_textura(self):
        rng = np.random.default_rng(1)
        ruido = cv2.GaussianBlur(rng.normal(128, 40, (240, 240, 3)).astype(np.float32), (0, 0), 1.2)
        foto = np.clip(ruido, 0, 255).astype(np.uint8)
        cv2.circle(foto, (120, 120), 80, (60, 60, 200), -1)
        mascara = np.full((240, 240), 255, np.uint8)
        self.assertGreater(indice_de_textura(foto, mascara), 0.45)
        lisa, m = self.dibujo()
        self.assertLess(indice_de_textura(lisa, m), 0.45)
        r = Foto3D(Parametros3D.preset("arcilla", lado_max=240, mascara_sujeto=np.full((240, 240), 255, np.uint8))).procesar(foto)
        self.assertTrue(any("textura" in a for a in r.advertencias))

    def test_elegir_preset_segun_la_textura(self):
        lisa, _ = self.dibujo()
        self.assertEqual(elegir_preset(lisa), "elementos")
        rng = np.random.default_rng(1)
        ruido = cv2.GaussianBlur(rng.normal(128, 40, (240, 240, 3)).astype(np.float32), (0, 0), 1.2)
        foto = np.clip(ruido, 0, 255).astype(np.uint8)
        cv2.circle(foto, (120, 120), 80, (60, 60, 200), -1)
        self.assertIn(elegir_preset(foto), ("elementos", "elementos_foto"))

    def test_parametros_de_estilo_invalidos(self):
        m = Foto3D()
        for malo in (dict(partes=1), dict(partes=40), dict(quitar_contornos=2), dict(aplanar_color=-1),
                     dict(mezcla_piezas=3), dict(brillo_suave=-1)):
            with self.assertRaises(ValueError, msg=str(malo)):
                m.procesar(imagen_prueba(), Parametros3D(**malo))


if __name__ == "__main__":
    unittest.main()
