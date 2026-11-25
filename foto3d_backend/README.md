# Foto3D: efecto 3D para cualquier imagen

Módulo de procesamiento de imágenes que convierte una foto común en una imagen con **look de render 3D**
(volumen, luz suave, brillo, grosor, sombras y, opcionalmente, un marco del que sale el sujeto).
Usa solo técnicas clásicas (sin redes neuronales). Es una aproximación 2.5D: no reconstruye geometría real.

## Instalación y uso rápido

```bash
pip install -r requirements.txt
python demo.py vaca.jpg                                   # efecto 3D, guarda foto_3d.png
python demo.py batido_3d.png --preset juguete               # más volumen y brillo (suave | juguete | intenso)
python demo.py  batido_3d.png --preset arcilla               # look de juguete 3D (dibujos de colores lisos)
python demo.py vaca.jpg  --preset diorama               # lo mismo, dentro de una escena con pared, piso y nubes
python demo.py 3d.png --marco                          # el sujeto sale de un marco
python demo.py foto.jpg --luz-x 0.5 --brillo 0.6 --etapas
python demo.py vaca_3d.png --objetos --sombra-piso 0.6   # cada objeto con su volumen y sombras entre planos
python -m unittest discover -s tests -v                  # 49 pruebas
```

```python
import cv2
from foto3d import Foto3D, Parametros3D

modulo = Foto3D(Parametros3D.preset("juguete"))   # o Parametros3D(inflado=0.8, ...)
r = modulo.procesar("foto.jpg")        # también acepta un arreglo NumPy BGR (cv2.imread)
cv2.imwrite("salida.png", r.imagen)

r.sujeto, r.fondo, r.mascara, r.alfa      # capas intermedias
r.mapa_altura, r.mapa_normales            # el volumen, para visualizarlo
r.tiempos, r.advertencias                 # segundos por etapa y avisos legibles
```

Contrato: **entrada** = imagen + `Parametros3D`; **salida** = `Resultado3D`. Un parámetro inválido lanza
`ValueError` con mensaje en español; una imagen ilegible lanza `FileNotFoundError` o `TypeError`.
Las imágenes usan la convención de OpenCV (BGR).

## Cómo funciona

| Etapa | Archivo | Técnica |
|---|---|---|
| 1. Separar el sujeto | `segmentacion.py` | Modelo del fondo desde el borde de la foto (ajuste polinómico + k-means) que inicializa GrabCut; limpieza morfológica y canal alfa suave con filtro guiado. |
| 2. Volumen | `volumen.py` | "Inflado" de la silueta. Perfil `globo` (por defecto): inflado de Poisson (laplaciano de u = -1, h = raíz de u), una cúpula con curvatura continua en todo el cuerpo, resuelta de grueso a fino con Jacobi. Perfil `borde`: transformada de distancia con borde de cuarto de círculo y meseta. A la altura se suma el relieve fino de la luminancia de la foto; las normales salen del gradiente (Sobel). |
| 3. Luz | `iluminacion.py` | Difusa (Lambert) + brillo especular (Blinn-Phong) + luz de contorno + oclusión de cavidades + saturación. |
| 4. Grosor y sombras | `sombras.py` | Extrusión por copias desplazadas del sujeto; sombra proyectada desenfocada más sombra de contacto, siempre del lado opuesto a la luz. Opcional: sombra de piso (silueta reflejada y aplanada, nítida en la base y más difusa al alejarse). |
| 5. Escena | `escena.py` | Fondo (original reconstruido con inpainting, desenfocado, degradado o color) y marco de madera con bisel iluminado; la barra inferior se dibuja delante del sujeto. |

## Modo por objetos (efecto diorama)

Con `Parametros3D(capas="objetos")` el sujeto se divide en objetos (componentes conectados; los que se tocan se
separan con watershed sobre los bordes de color, usando como marcadores los picos de la transformada de
distancia). Cada objeto recibe su **propio volumen y luz**, y se dibuja del más lejano al más cercano (se
estima por la posición de su base: lo que apoya más abajo está más cerca). Cada uno proyecta su sombra sobre el
fondo y sobre los objetos que tiene detrás, que es lo que da la sensación de planos separados.
`sombra_piso` agrega la sombra proyectada en el piso para los objetos apoyados en la parte baja de la foto.

## Estilo arcilla: el efecto de las imágenes de referencia

Los renders de juguete 3D (las sandías, el aguacate) se ven distintos de un dibujo con sombreado encima porque no
tienen contornos negros ni texturas pintadas: cada parte es un volumen de color liso con su propio brillo. El perfil
`piezas` reproduce eso en cuatro pasos:

1. **Quitar los contornos** (`quitar_contornos`): se detectan las líneas oscuras largas con un black-hat morfológico y
   se rellenan con el color vecino. Los detalles cortos (boca, pestañas, semillas) se conservan.
2. **Aplanar los colores** (`aplanar_color`): filtro bilateral repetido y desplazamiento de media, para tener zonas de
   color liso.
3. **Separar en piezas** (`partes`): k-means en el espacio Lab + componentes conectados; las piezas diminutas se unen a
   la vecina. Cada pieza se abomba por separado con un inflado de Poisson, y entre piezas queda un surco.
4. **Iluminar como plástico**: reflejo especular agudo más un reflejo ancho y tenue (`brillo_suave`), luz de contorno y
   sombras proyectadas.

Se usa con los presets `arcilla` (solo el objeto) y `diorama` (el objeto dentro de una escena con pared, piso y nubes).

| Tipo de imagen | Preset recomendado |
|---|---|
| Dibujo o ilustración de colores lisos | `arcilla` o `diorama` |
| Foto o render con mucho detalle | `suave`, `juguete` (o `juguete` con `fondo="escena"`) |

El estilo de piezas con fotos detalladas se ve "derretido", porque trata cada detalle como una pieza. El módulo mide la
textura de la imagen y, si es alta, lo avisa en `advertencias`. Si las piezas salen demasiado fragmentadas, bajá
`partes`; si el volumen se ve muy "acolchado", bajá `mezcla_piezas`.

## Presets y cómo lograr más efecto 3D

Si el resultado se ve muy plano (suele pasar con dibujos y figuras de colores lisos), subí el preset:

| Preset | Qué da |
|---|---|
| `suave` (por defecto) | Volumen y brillo moderados; funciona bien con fotos. |
| `juguete` | Más relieve y brillo; ideal para dibujos, íconos y personajes. |
| `arcilla` | Look de juguete 3D para dibujos de colores lisos: sin contornos, piezas de plástico brillante. |
| `diorama` | `arcilla` dentro de una escena: pared y piso de colores lisos, nubes con volumen, sombra en el piso. Agranda el lienzo. |
| `intenso` | Contraste fuerte de luz y sombra; puede oscurecer detalles del lado opuesto a la luz. |

```python
Foto3D(Parametros3D.preset("juguete")).procesar("dibujo.png")
Foto3D(Parametros3D.preset("juguete", luz_x=0.5)).procesar("dibujo.png")   # y cambiar un valor
```

El perfil `globo` es lo que da la sensación de cuerpo redondo; el perfil `borde` redondea solo el contorno y deja el
interior plano (es el que se usa para el marco de madera).

## Escena de diorama

Con `fondo="escena"` (o el preset `diorama`) el sujeto se coloca en una escena como la de las imágenes de referencia:
una pared y un piso de colores lisos separados por un horizonte inclinado, nubes decorativas con volumen y la sombra
del sujeto proyectada en el piso. El lienzo se agranda para dejar lugar a la escena.

```python
Foto3D(Parametros3D.preset("diorama")).procesar("dibujo.png")
Foto3D(Parametros3D.preset("diorama", nubes=5, color_pared=(170, 230, 250), color_piso=(230, 170, 200))).procesar("dibujo.png")
```

Los colores van en BGR (el orden de OpenCV). Las nubes se ubican sobre la pared sin tocar al sujeto: si no hay lugar
para todas las que se piden, se achican o se omiten (`nubes` es un máximo).

## Parámetros principales (`Parametros3D`)

| Parámetro | Por defecto | Qué cambia |
|---|---|---|
| `perfil` | `"globo"` | `globo` (curvatura continua), `piezas` (cada pieza de color abombada) o `borde` (solo el contorno). |
| `partes`, `mezcla_piezas` | 8, 0.5 | Perfil `piezas`: colores usados para separar piezas y cuánto pesa cada pieza frente al cuerpo. |
| `quitar_contornos`, `aplanar_color` | 0.0, 0.0 | Borrar contornos negros y aplanar colores (0 a 1). |
| `brillo_suave` | 0.25 | Reflejo ancho y tenue (aspecto de plástico pulido). |
| `inflado` | 0.8 | Altura del volumen (0 casi plano, 1 muy abombado). |
| `relieve`, `detalle` | 1.5, 0.25 | Fuerza del relieve y cuánto marca la foto los detalles finos. |
| `luz_x`, `luz_y`, `luz_z` | -0.5, -0.6, 0.8 | Dirección de la luz (por defecto arriba a la izquierda). |
| `ambiente`, `brillo`, `dureza_brillo` | 0.4, 0.5, 40 | Luz base (menor = más contraste) y reflejo especular. | Reflejo tipo plástico o arcilla brillante. |
| `grosor` | 0.012 | Grosor de la figura recortada (fracción del lado mayor). |
| `sombra_opacidad`, `sombra_desplazamiento`, `sombra_desenfoque` | 0.45, 0.03, 0.02 | Sombra proyectada. |
| `fondo` | `"auto"` | `original`, `desenfocado`, `degradado`, `color` o `escena` (`auto` usa degradado si hay marco). |
| `color_pared`, `color_piso` | celeste, coral | Colores BGR de la escena (`fondo="escena"`). |
| `nubes`, `horizonte`, `inclinacion_horizonte` | 0, 0.55, 0.12 | Cantidad de nubes (0 a 6), altura y pendiente del piso. |
| `marco`, `sobresale`, `bajo_marco` | `False`, 0.22, 0.05 | Efecto "sale del marco". |
| `capas` | `"una"` | `"objetos"` activa el efecto diorama. |
| `separar_tocados`, `sensibilidad_separacion` | `True`, 0.6 | Dividir objetos pegados; menor valor separa más (0.4 separa piezas de distinto tamaño, pero puede cortar personajes). |
| `sombra_piso` | 0.0 | Sombra en el piso (0 a 1). |
| `quitar_sombras` | `False` | Trata las sombras que ya trae la foto como fondo. |
| `mascara_sujeto` | `None` | Máscara propia (blanco = sujeto) si la detección automática falla. |

## Límites conocidos

- La detección automática supone que **el borde de la foto es fondo**. Funciona muy bien con fondos lisos,
  degradados o de pocos colores planos. Con fondos desordenados puede equivocarse; en ese caso hay que pasar
  `mascara_sujeto`. Si no puede separar nada, usa una elipse central y lo avisa en `advertencias`.
- Las sombras que ya existen en la foto original pueden quedar incluidas en el sujeto; `quitar_sombras=True` las trata como fondo, pero falla con objetos grises sobre fondo blanco (se confunden con una sombra).
- Si la imagen ya trae luces y sombras de un render 3D, el efecto se suma a ese sombreado. Donde mejor funciona es con fotos e ilustraciones planas.
- En modo objetos, piezas pegadas de distinto tamaño pueden no separarse bien; se ajusta con `sensibilidad_separacion` o con una máscara propia.
- Es un efecto de relieve y luz, no una reconstrucción: se ve mejor con sujetos de formas redondeadas.
