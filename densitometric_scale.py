import argparse
import random
import string
import os
from PIL import Image, ImageDraw, ImageFont

class DensitometricScaleGenerator:
    """
    Generador de escalas densitométricas para calibración y pruebas de contraste.

    Esta clase permite crear imágenes (PNG) que contienen una serie de parches con 
    diferentes niveles de densidad, organizados en filas y columnas, con leyendas
    informativas y espaciadores de control para espectrofotometría.

    La conversión de densidad (D) a valor de píxel (V) sigue la fórmula:
    V = 255 * (10^-D)^(1/gamma)

    Parámetros de la Clase (API):
        - dpi (int): Resolución de salida (default: 300).

    Parámetros del método create_scale_image (API):
        - density_step (float): Incremento de densidad (ej. 0.1, 0.05).
        - max_density (float): Límite superior de densidad (ej. 2.5).
        - rows (int): Forzar número de filas (debe ser divisor del total de parches).
        - paper_size (str): 'A4' o 'A3' (en orientación portrait).
        - patch_size_mm (float): Tamaño del parche (default: 7.0mm).
        - spacer_size_mm (float): Tamaño del espaciador negro/blanco (default: 2.0mm).
        - gamma (float): Factor de corrección gamma (default: 2.2).
        - output_path (str): Nombre del archivo de salida.

    Parámetros del CLI:
        --step: Incremento de densidad.
        --max_d: Densidad máxima.
        --rows: Número de filas.
        --paper: 'A4' o 'A3'.
        --patch_size: Tamaño de parche en mm.
        --gamma: Factor gamma.
        --output: Archivo de salida.
    """
    
    PAPER_SIZES = {
        'A4': (210, 297),
        'A3': (297, 420)
    }
    
    def __init__(self, dpi=300):
        self.dpi = dpi
        self.mm_to_px_factor = dpi / 25.4
        
    def mm_to_px(self, mm):
        return int(round(mm * self.mm_to_px_factor))
    
    def calculate_pixel_value(self, density, gamma=2.2):
        """
        Calcula el valor de píxel (0-255) basado en la densidad y gamma.
        Fórmula: 255 * (10^-D)^(1/gamma)
        """
        return int(round(255 * (10**(-density))**(1/gamma)))

    def generate_random_ref(self, length=5):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

    def create_scale_image(self, 
                           density_step=0.1, 
                           max_density=2.5,
                           rows=None, 
                           paper_size='A4', 
                           patch_size_mm=7.0, 
                           spacer_size_mm=2.0,
                           gamma=2.2,
                           output_path="densitometric_scale.png"):
        """
        Genera una imagen con la escala densitométrica basada en los parámetros indicados.

        Args:
            density_step (float): Incremento de densidad entre parches (ej. 0.1, 0.05).
            max_density (float): Valor máximo de densidad hasta el que llegará la escala.
            rows (int, opcional): Número fijo de filas. Si es None, se calcula automáticamente
                                  asegurando que todas las filas tengan el mismo número de columnas.
            paper_size (str): Tamaño del soporte de impresión ('A4' o 'A3').
            patch_size_mm (float): Dimensión del parche cuadrado (default 7mm).
            spacer_size_mm (float): Ancho total del espaciador (1mm negro + 1mm blanco = 2mm).
            gamma (float): Factor gamma para la conversión de densidad a píxel (default 2.2).
            output_path (str): Nombre o ruta del archivo de imagen de salida.

        Returns:
            str: Ruta del archivo generado.

        Raises:
            ValueError: Si la configuración de parches no cabe en el papel o si el número total
                        de parches no es divisible equitativamente entre las filas.
        """
        
        # 1. Configuración de dimensiones del papel
        if paper_size.upper() not in self.PAPER_SIZES:
            raise ValueError(f"Tamaño de papel no soportado: {paper_size}. Use A4 o A3.")
        
        width_mm, height_mm = self.PAPER_SIZES[paper_size.upper()]
        img_w = self.mm_to_px(width_mm)
        img_h = self.mm_to_px(height_mm)
        
        # Margenes (2cm L/R, 4cm T/B)
        margin_lr_px = self.mm_to_px(20)
        margin_tb_px = self.mm_to_px(40)
        
        available_width_px = img_w - (2 * margin_lr_px)
        
        # 2. Preparar parches
        densities = []
        d = 0.0
        while d <= max_density + 0.0001: # Pequeña tolerancia para flotantes
            densities.append(round(d, 3))
            d += density_step
            
        num_patches = len(densities)
        
        # Dimensiones de parches y espaciadores en px
        patch_px = self.mm_to_px(patch_size_mm)
        spacer_px = self.mm_to_px(spacer_size_mm)
        half_spacer_px = self.mm_to_px(spacer_size_mm / 2) # 1mm black, 1mm white
        
        # 3. Calcular Layout (Columnas y Filas)
        # Ancho fila = (C * patch_px) + ((C + 1) * spacer_px)
        
        def get_row_width(cols):
            return (cols * patch_px) + ((cols + 1) * spacer_px)
        
        # Determinar número máximo de columnas que caben físicamente
        max_cols_physical = 0
        for c in range(1, num_patches + 1):
            if get_row_width(c) <= available_width_px:
                max_cols_physical = c
            else:
                break
        
        if max_cols_physical == 0:
            raise ValueError("El tamaño del parche/espaciador es demasiado grande para el papel.")

        if rows is None:
            # Modo Automático: Buscar el mejor número de columnas que divida el total de parches
            # y que quepa en el ancho del papel.
            if num_patches <= max_cols_physical:
                cols = num_patches
                rows = 1
            else:
                # Buscar un divisor de num_patches que sea <= max_cols_physical
                possible_cols = [c for c in range(1, max_cols_physical + 1) if num_patches % c == 0]
                if not possible_cols or (len(possible_cols) == 1 and possible_cols[0] == 1 and num_patches > 1):
                    # Si es primo o no hay divisores que quepan bien, lanzamos error como pide el usuario
                    # ya que no se pueden hacer filas iguales.
                    raise ValueError(f"No se puede organizar {num_patches} parches en filas iguales que quepan en el ancho del papel ({max_cols_physical} columnas máx). "
                                     f"El número total de parches debe ser divisible por el número de columnas.")
                cols = max(possible_cols)
                rows = num_patches // cols
        else:
            # Validar si las columnas caben
            if num_patches % rows != 0:
                 raise ValueError(f"No se puede organizar {num_patches} parches en {rows} filas iguales. "
                                  f"El número total de parches ({num_patches}) debe ser divisible por el número de filas.")
            cols = num_patches // rows
            if get_row_width(cols) > available_width_px:
                raise ValueError(f"La configuración de {cols} columnas excede el ancho disponible del papel ({width_mm}mm menos márgenes). "
                                 f"Máximo permitido: {max_cols_physical} columnas.")

        # 4. Crear Lienzo
        image = Image.new('L', (img_w, img_h), color=255) # 'L' para escala de grises, 255=blanco
        draw = ImageDraw.Draw(image)
        
        # Intentar cargar una fuente, si no cargar default
        try:
            # Fuentes comunes en Windows
            font_size = self.mm_to_px(3)
            font = ImageFont.truetype("arial.ttf", font_size)
            small_font = ImageFont.truetype("arial.ttf", self.mm_to_px(2))
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # 5. Escribir metadatos (margen superior)
        ref_id = self.generate_random_ref()
        metadata_text = f"REF: {ref_id} | STEP: {density_step} | GAMMA: {gamma} | PATCHES: {num_patches} | ROWS: {rows} | COLS: {cols}"
        draw.text((margin_lr_px, margin_tb_px - self.mm_to_px(10)), metadata_text, fill=0, font=font)
        
        # 6. Dibujar la escala
        row_height_px = patch_px + self.mm_to_px(15) # Espacio para el parche + leyenda debajo
        
        current_patch_idx = 0
        for r in range(rows):
            y_start = margin_tb_px + (r * row_height_px)
            
            # Dibujar espaciadores y parches en la fila
            for c in range(cols):
                if current_patch_idx >= num_patches:
                    break
                
                # Dibujar espaciador ANTES del parche
                x_spacer = margin_lr_px + (c * (patch_px + spacer_px))
                x_patch = x_spacer + spacer_px
                
                # Espaciador: 1mm negro, 1mm blanco
                draw.rectangle([x_spacer, y_start, x_spacer + half_spacer_px, y_start + patch_px], fill=0)
                draw.rectangle([x_spacer + half_spacer_px, y_start, x_spacer + spacer_px, y_start + patch_px], fill=255)
                
                # Dibujar parche
                density = densities[current_patch_idx]
                pixel_val = self.calculate_pixel_value(density, gamma)
                draw.rectangle([x_patch, y_start, x_patch + patch_px, y_start + patch_px], fill=pixel_val)
                
                # Dibujar Leyenda debajo del parche
                legend_y = y_start + patch_px + self.mm_to_px(1)
                d_text = f"D:{density:.2f}"
                p_text = f"V:{pixel_val}"
                
                draw.text((x_patch, legend_y), d_text, fill=0, font=small_font)
                draw.text((x_patch, legend_y + self.mm_to_px(3)), p_text, fill=0, font=small_font)
                
                # Si es el último parche de la fila O el último parche total, dibujar el espaciador final
                if c == cols - 1 or current_patch_idx == num_patches - 1:
                    x_final_spacer = x_patch + patch_px
                    draw.rectangle([x_final_spacer, y_start, x_final_spacer + half_spacer_px, y_start + patch_px], fill=0)
                    draw.rectangle([x_final_spacer + half_spacer_px, y_start, x_final_spacer + spacer_px, y_start + patch_px], fill=255)

                current_patch_idx += 1

        # 7. Guardar
        image.save(output_path, dpi=(self.dpi, self.dpi))
        print(f"Imagen guardada en: {output_path}")
        print(f"Configuración: {rows} filas, {cols} columnas, {num_patches} parches.")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Densitometric Scale Generator")
    parser.add_argument("--step", type=float, default=0.1, help="Density increment (e.g., 0.1)")
    parser.add_argument("--max_d", type=float, default=2.5, help="Maximum density")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows (optional)")
    parser.add_argument("--paper", type=str, default="A4", choices=["A4", "A3"], help="Paper size")
    parser.add_argument("--patch_size", type=float, default=7.0, help="Patch size in mm")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma for conversion")
    parser.add_argument("--output", type=str, default="escala_densidad.png", help="Output filename")
    
    args = parser.parse_args()
    
    gen = DensitometricScaleGenerator()
    try:
        gen.create_scale_image(
            density_step=args.step,
            max_density=args.max_d,
            rows=args.rows,
            paper_size=args.paper,
            patch_size_mm=args.patch_size,
            gamma=args.gamma,
            output_path=args.output
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
