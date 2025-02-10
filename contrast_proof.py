from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import math
import io

from colormath.color_objects import HSVColor, sRGBColor
from colormath.color_conversions import convert_color

import time


class CurveVsHisto:

    def __init__(self,lista, image_path, paper_name ):

        lista = [float(s) for s in lista.split(',')]
        lista = self.normalizeShadows(lista)
        curve = self.normalizar_curva(lista)

        self.paper_name = paper_name.replace(" ","_")
        cv_img = cv.imread(image_path)

        hist_buf, falsecolor, step = self.histograma_cv2(cv_img, curve)
        falsecolor_img = self.remplaceColors(cv_img, falsecolor, step)
        self.mostrarGraficos(hist_buf, falsecolor_img)

    def normalizar_curva(self, curva, num_puntos=20):

        curva = np.array(curva, dtype=float)
        n = len(curva)
        if n == 0:
            raise ValueError("La curva no puede estar vacía.")
        indices_originales = np.linspace(0, n - 1, num=n)
        nuevos_indices = np.linspace(0, n - 1, num=num_puntos)
        curva_normalizada = np.interp(nuevos_indices, indices_originales, curva)
        return curva_normalizada.tolist()

    def histograma_cv2(self, img, curve):

        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Procesar la curva
        curve = np.sort(curve)
        min_val = np.amin(curve)
        max_val = np.amax(curve)
        step = (max_val - min_val) / len(curve)

        # Se genera el vector t y se calculan las diferencias normalizadas
        t = np.arange(min_val, max_val, step)
        dx = np.diff(curve) / step
        dx /= np.max(np.abs(dx))

        # Calculamos las zonas de recorte inferior y superior
        inf = min_val / step
        sup = (255 - max_val) / step
        fillXinf = [min_val - step * z for z in range(int(math.ceil(inf)) + 1)]
        fillXsup = [max_val + step * y for y in range(int(math.ceil(sup)) + 1)]

        # Concatenamos los valores para el eje X
        x = fillXinf + t.tolist() + fillXsup
        y_vals = [255] * len(x)
        d = [dx[0]] + dx.tolist()

        falsecolor = []
        fig, ax1 = plt.subplots()

        viridis = plt.get_cmap('plasma', 256)

        # Construir la lista de colores y dibujar las barras
        x_values = []
        colors = []
        for i, x_val in enumerate(x):
            if i < inf + 1:
                rgb = (0.3, 0.3, 0.3)
            elif i > len(d) + inf + 1:
                rgb = (0.6, 0.6, 0.6)
            else:
                idx = int(i - inf - 1)
                idx = max(0, min(idx, len(d) - 1))
                rgb = viridis(d[idx])
            falsecolor.append((x_val, rgb))
            x_values.append(x_val)
            colors.append(rgb)

        # Dibujar las barras del histograma (con colores asignados)
        ax1.bar(x_values, y_vals, color=colors, width=step)

        # Agregar una colorbar
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='plasma'), ax=ax1)
        cbar.set_ticks([])
        ax_cbar = cbar.ax
        ax_cbar.text(1.5, 0.5, 'Contrast', rotation=90)

        # Usar un segundo eje para graficar el histograma de los valores de píxeles
        ax2 = ax1.twinx()
        ax2.tick_params(axis='both', which='both', length=0)
        ax2.set_yticklabels([])
        ax2.set_xlabel('Pixel Value')

        # Obtener los datos de píxeles a partir de la imagen en escala de grises
        a = gray.ravel()
        ax2.hist(a, bins=255, color='gray', alpha=0.5)

        # Dibujar la curva sobre el histograma
        ax3 = ax2.twinx()
        ax3.plot(t, curve, color='tab:blue', label="Paper Curve")
        ax3.set_yticklabels([])

        # Guardar la figura en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf, falsecolor, step

    def remplaceColors(self, image, lut, step):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        for i in lut:
            lo = np.array([0, 0, i[0]])
            hi = np.array([255, 255, i[0] + step])
            mask = cv.inRange(hsv, lo, hi)
            rgb = sRGBColor(i[1][0], i[1][1], i[1][2])
            hsv_values = convert_color(rgb, HSVColor, target_illuminant='d50')
            image[mask > 0] = ((hsv_values.hsv_h / 360) * 180, hsv_values.hsv_s * 255, hsv_values.hsv_v * 255)

        rgb = cv.cvtColor(image, cv.COLOR_HSV2BGR)
        rgb = self.resize_image_by_width(rgb)

        return rgb

    def mostrarGraficos(self, histo_buf, falsecolor_img):
        # Crear la figura con dos subplots para la imagen del histograma y la imagen falseada.
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Abrir la imagen del histograma desde el buffer.
        histo_img = Image.open(histo_buf)

        # Primer subplot: Histograma con el rótulo (paper_name)
        ax[0].imshow(histo_img)
        ax[0].set_title("Histogram - " + self.paper_name)
        ax[0].axis("off")

        # Segundo subplot: Imagen en falso color con el rótulo
        ax[1].imshow(cv.cvtColor(falsecolor_img, cv.COLOR_BGR2RGB))
        ax[1].set_title("False Color - " + self.paper_name)
        ax[1].axis("off")

        # Figura adicional para guardar el histograma
        fig_hist, ax_hist = plt.subplots(figsize=(6, 6))
        ax_hist.imshow(histo_img)
        ax_hist.set_title("Histogram - " + self.paper_name)
        ax_hist.axis("off")

        path_chart = 'output/histogram_'+self.paper_name+'.png'
        path_false_color = 'output/false_color_'+self.paper_name+'.png'
        fig_hist.savefig(path_chart, format='png', bbox_inches='tight')
        plt.close(fig_hist)
        cv.imwrite(path_false_color, falsecolor_img)

        self.concatenar_imagenes_horizontal(path_chart,path_false_color )

        plt.show()

    def concatenar_imagenes_horizontal(self, ruta_imagen1, ruta_imagen2):

        img1 = cv.imread(ruta_imagen1)
        img2 = cv.imread(ruta_imagen2)

        # Verificar que se hayan cargado correctamente
        if img1 is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen1}")
        if img2 is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen2}")

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 != h2:
            ratio = h1 / h2
            nuevo_ancho = int(w2 * ratio)
            img2 = cv.resize(img2, (nuevo_ancho, h1), interpolation=cv.INTER_AREA)

        imagen_concatenada = cv.hconcat([img1, img2])

        cv.imwrite('output/proof_'+self.paper_name+'.png', imagen_concatenada)

        return imagen_concatenada
    def resize_image_by_width(self, image, target_width=600):

        original_height, original_width = image.shape[:2]
        if original_width <= target_width:
            return image
        aspect_ratio = original_height / original_width
        new_height = int(target_width * aspect_ratio)
        resized_image = cv.resize(image, (target_width, new_height), interpolation=cv.INTER_AREA)
        return resized_image

    def normalizeShadows(self, curve):
        # elimina información en las sombras para evitar ruido de la lectura
        # si detecta 3 valores seguiros con el mismo nivel recorta la curva en esa zona

        for i in range(len(curve) - 3):
            if curve[i] - curve[i + 3] == 0:
                return curve[:max(i - 2, 0)]

        return curve


if __name__ == '__main__':

    inicio = time.time()

    demo_curve = "227.169,215.685,203.8,188.167,170.234,153.604,136.616,119.722,103.842,89.027,75.912,64.3,54.322,49.044,41.771,37.202,36,35.8,35.6,35.5"
    demo_curve = "202,196,188,180,173,166,160,154,146,140,136,129,123,118,112,108,104,99,95,90,86,83,79,76,73,70,67,65,63,60,59,57,56,54,53,51"

    demo_path = "test-imgs/upzji_dsc_0056.tiff"

    paper_name = "Hanemule"

    cvh = CurveVsHisto(demo_curve, demo_path, paper_name)


    fin = time.time()
    print(fin - inicio)
