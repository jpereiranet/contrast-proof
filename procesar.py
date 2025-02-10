import csv
import os
from contrast_proof import CurveVsHisto  # Asegúrate de importar la clase desde el módulo adecuado


def procesar_csv_y_aplicar_clase(csv_file):

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # next(reader, None)

        for row in reader:
            paper_name = row[0].strip()
            curve_values = row[1].strip()

            print(f"Procesando paper: {paper_name}")

            image_path = "test-imgs/upzji_dsc_0056.tiff"

            CurveVsHisto(curve_values, image_path, paper_name)

if __name__ == '__main__':

    csv_file = 'papers.csv'
    procesar_csv_y_aplicar_clase(csv_file)
