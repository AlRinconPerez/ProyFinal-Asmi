import numpy as np
import pandas as pd
import os
from base_extractor import BaseGlobalFeaturesExtractor
import re  # Para usar expresiones regulares

def obtener_informacion_usuario(nombre_archivo):
    """
    Esta función extrae el número de usuario, el dígito escrito, la sesión y la muestra
    del nombre del archivo.
    """
    match = re.match(r'u(\d+)_digit_(\d+)_(\d+).txt', nombre_archivo)
    
    if match:
        usuario = match.group(1)  # Número de usuario
        digito = match.group(2)  # Dígito escrito
        muestra_sesion = int(match.group(3))  # Número de sesión/muestra

        if muestra_sesion <= 8:
            sesion = 1
            muestra = muestra_sesion // 2  
        else:
            sesion = 2
            muestra = (muestra_sesion - 8) // 2 

        return usuario, digito, sesion, muestra

    return None, None, None, None

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    proy_asmi_dir = os.path.dirname(os.path.dirname(script_dir))
    carpeta = os.path.join(proy_asmi_dir, "e-BioDigit_DB")

    archivos = []
    for root, _, files in os.walk(carpeta):
        for file in files:
            archivos.append(os.path.join(root, file))

    base_extractor = BaseGlobalFeaturesExtractor()
    df_list = []

    for input_filename in archivos:
        print(f"Procesando archivo: {input_filename}")

        try:
            data = np.loadtxt(input_filename, skiprows=1)
            x = data[:, 0]
            y = data[:, 1]
            timestamp = data[:, 2]

            base_features = base_extractor.extract(x, y, timestamp)

            output_csv = os.path.basename(input_filename).replace('.txt', '.csv')
            np.savetxt(output_csv, base_features, fmt='%f', delimiter=",")

            df = pd.read_csv(output_csv, header=None)
            df.insert(0, "car_id", range(0, len(df)))
            df.columns = ["car_id", "valor"]
            df["label"] = "Car." + df["car_id"].astype(str)

            usuario_num, digito, sesion, muestra = obtener_informacion_usuario(os.path.basename(input_filename))

            df["usuario"] = os.path.basename(input_filename)  # Nombre completo
            df["usuario_num"] = usuario_num
            df["digito"] = digito
            df["sesion"] = sesion
            df["muestra"] = muestra

            dataset = df.pivot(index='usuario', columns='label', values='valor')
            dataset = dataset.reindex(sorted(dataset.columns, key=lambda x: int(x.split('.')[1])), axis=1)

            dataset.insert(0, "usuario", os.path.basename(input_filename))
            dataset.insert(1, "usuario_num", usuario_num)
            dataset.insert(2, "digito", digito)
            dataset.insert(3, "sesion", sesion)
            dataset.insert(4, "muestra", muestra)

            df_list.append(dataset)
            os.remove(output_csv)

        except Exception as e:
            print(f"Error procesando el archivo {input_filename}: {e}")

    final_df = pd.concat(df_list, axis=0)

    final_df.to_csv("archivo_modificado_final.csv", index=False, na_rep="nan")

    print("Se ha generado el archivo con el orden de columnas correcto.")
