import pandas as pd

# Diccionario de códigos de países a nombres completos
country_dict = {
    'AT': 'Austria',
    'BE': 'Belgium',
    'BG': 'Bulgaria',
    'CY': 'Cyprus',
    'CZ': 'Czechia',
    'DE': 'Germany',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'EL': 'Greece',
    'ES': 'Spain',
    'EU27_2020': 'European Union (27 countries - 2020)',
    'EU28': 'European Union (28 countries)',
    'FI': 'Finland',
    'FR': 'France',
    'HR': 'Croatia',
    'HU': 'Hungary',
    'IE': 'Ireland',
    'IS': 'Iceland',
    'IT': 'Italy',
    'LI': 'Liechtenstein',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'LV': 'Latvia',
    'ME': 'Montenegro',
    'MK': 'North Macedonia',
    'MT': 'Malta',
    'NL': 'Netherlands',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'RS': 'Serbia',
    'SE': 'Sweden',
    'SI': 'Slovenia',
    'SK': 'Slovakia',
    'TR': 'Turkey',
    'UK': 'United Kingdom',
    'XK': 'Kosovo',
    'BA': 'Bosnia and Herzegovina',
    'AL': 'Albania'
}

# Lista de países a conservar
countries_to_keep = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
    'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Lithuania',
    'Luxembourg', 'Malta', 'Netherlands', 'Norway', 'Poland',
    'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'    
]

def load_csv_data(filepath):
    """
    Carga datos desde un archivo CSV.
    
    Parámetros:
    filepath (str): La ruta del archivo CSV que se desea cargar.
    
    Retorna:
    pd.DataFrame: Un DataFrame con los datos cargados del archivo CSV.
    """
    return pd.read_csv(filepath)

def load_excel_data(filepath, sheet_name):
    """
    Carga datos desde un archivo Excel.
    
    Parámetros:
    filepath (str): La ruta del archivo Excel que se desea cargar.
    sheet_name (str): El nombre de la hoja de cálculo en el archivo Excel.
    
    Retorna:
    pd.DataFrame: Un DataFrame con los datos cargados de la hoja de cálculo especificada.
    """
    return pd.read_excel(filepath, engine='openpyxl', sheet_name=sheet_name)

def filter_by_year(df, start_year, end_year):
    """
    Filtra el DataFrame por el rango de años dado.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    start_year (int): El año inicial del rango de filtrado.
    end_year (int): El año final del rango de filtrado.
    
    Retorna:
    pd.DataFrame: El DataFrame filtrado para el rango de años especificado.
    """
    return df[(df['TIME_PERIOD'] <= end_year) & (df['TIME_PERIOD'] >= start_year)]

def select_columns(df, columns):
    """
    Selecciona columnas específicas del DataFrame.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame del que se seleccionarán las columnas.
    columns (list): Una lista de nombres de columnas a seleccionar.
    
    Retorna:
    pd.DataFrame: Un DataFrame con solo las columnas seleccionadas.
    """
    return df[columns]

def rename_columns(df, new_column_names):
    """
    Renombra las columnas del DataFrame.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame con las columnas a renombrar.
    new_column_names (list): Una lista con los nuevos nombres de las columnas.
    
    Retorna:
    pd.DataFrame: El DataFrame con las columnas renombradas.
    """
    df.columns = new_column_names
    return df

def replace_country_codes(df, country_dict):
    """
    Reemplaza los códigos de países con nombres completos.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene la columna de códigos de países.
    country_dict (dict): Un diccionario que mapea códigos de países a nombres completos.
    
    Retorna:
    pd.DataFrame: El DataFrame con los códigos de países reemplazados por nombres completos.
    """
    df['Country'] = df['Country'].replace(country_dict)
    return df

def filter_countries(df, countries_to_keep):
    """
    Filtra el DataFrame para conservar solo los países listados.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    countries_to_keep (list): Una lista de nombres de países que se deben conservar.
    
    Retorna:
    pd.DataFrame: El DataFrame filtrado para conservar solo los países especificados.
    """
    return df[df['Country'].isin(countries_to_keep)]

def sort_dataframe(df, sort_columns):
    """
    Ordena el DataFrame por las columnas especificadas.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame a ordenar.
    sort_columns (list): Una lista de nombres de columnas por las que se ordenará el DataFrame.
    
    Retorna:
    pd.DataFrame: El DataFrame ordenado por las columnas especificadas.
    """
    return df.sort_values(by=sort_columns)

def reset_dataframe_index(df):
    """
    Reinicia el índice del DataFrame.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame cuyo índice se desea reiniciar.
    
    Retorna:
    pd.DataFrame: El DataFrame con el índice reiniciado.
    """
    return df.reset_index(drop=True)

def melt_dataframe(df, id_vars, var_name, value_name):
    """
    Convierte el DataFrame al formato largo.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame a convertir.
    id_vars (list): Una lista de nombres de columnas que se mantendrán como identificadores.
    var_name (str): El nombre de la columna que contendrá los nombres de las columnas originales.
    value_name (str): El nombre de la columna que contendrá los valores correspondientes.
    
    Retorna:
    pd.DataFrame: El DataFrame en formato largo.
    """
    return df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)

def process_recycling_data(filepath):
    """
    Procesa los datos de tasas de reciclaje.
    
    Parámetros:
    filepath (str): La ruta del archivo CSV que contiene los datos de tasas de reciclaje.
    
    Retorna:
    pd.DataFrame: El DataFrame procesado con tasas de reciclaje, filtrado, renombrado y ordenado.
    """
    df = load_csv_data(filepath)
    df = filter_by_year(df, 2013, 2022)
    df = select_columns(df, ['geo', 'TIME_PERIOD', 'OBS_VALUE'])
    df = rename_columns(df, ['Country', 'Year', 'recycling_rate'])
    df = replace_country_codes(df, country_dict)
    df = filter_countries(df, countries_to_keep)
    df = sort_dataframe(df, ['Country', 'Year'])
    df = reset_dataframe_index(df)
    return df

def process_waste_generation_data(filepath):
    """
    Procesa los datos de generación de residuos.
    
    Parámetros:
    filepath (str): La ruta del archivo CSV que contiene los datos de generación de residuos.
    
    Retorna:
    pd.DataFrame: El DataFrame procesado con datos de generación de residuos, filtrado, renombrado y agrupado.
    """
    df = load_csv_data(filepath)
    df = filter_by_year(df, 2013, 2022)
    df = df[df['waste'] == 'TOTAL']
    df = select_columns(df, ['geo', 'TIME_PERIOD', 'OBS_VALUE'])
    df = rename_columns(df, ['Country', 'Year', 'Total_waste'])
    df = replace_country_codes(df, country_dict)
    df = filter_countries(df, countries_to_keep)
    df = sort_dataframe(df, ['Country', 'Year'])
    df = reset_dataframe_index(df)
    df = df.groupby(['Country', 'Year'], as_index=False)['Total_waste'].sum()
    return df

def process_co2_emissions(filepath):
    """
    Procesa los datos de emisiones de CO2.
    
    Parámetros:
    filepath (str): La ruta del archivo Excel que contiene los datos de emisiones de CO2.
    
    Retorna:
    pd.DataFrame: El DataFrame procesado con datos de emisiones de CO2 en formato largo, filtrado y ordenado.
    """
    
    sheet_name='fossil_CO2_totals_by_country'
    df = load_excel_data(filepath, sheet_name)
    df = df.replace({
        'Spain and Andorra': 'Spain',
        'Italy, San Marino and the Holy See': 'Italy',
        'France and Monaco': 'France'
    })
    df.columns = df.columns.astype(str)
    columnas_interes = ['Country'] + [str(year) for year in range(2013, 2023)]
    df = select_columns(df, columnas_interes)
    df = melt_dataframe(df, id_vars=['Country'], var_name='Year', value_name='CO2_Emissions')
    df = replace_country_codes(df, country_dict)
    df = filter_countries(df, countries_to_keep)
    df = sort_dataframe(df, ['Country', 'Year'])
    df = reset_dataframe_index(df)
    df['Year'] = df['Year'].astype(int)
    return df

def process_pollution_level(filepath):
    """
    Procesa los datos de niveles de contaminación del aire.
    
    Parámetros:
    filepath (str): La ruta del archivo Excel que contiene los datos de contaminación del aire.
    
    Retorna:
    pd.DataFrame: El DataFrame procesado con niveles de contaminación del aire en formato largo, filtrado y ordenado.
    """
    sheet_name='Sheet 1'
    df = load_excel_data(filepath, sheet_name)
    df = df.rename(columns={'Country1.1': 'Country'})
    df.columns = df.columns.astype(str)
    columnas_interes = ['Country'] + [str(year) for year in range(2013, 2023)]
    df = select_columns(df, columnas_interes)
    df = melt_dataframe(df, id_vars=['Country'], var_name='Year', value_name='Air_pollution_level')
    df = filter_countries(df, countries_to_keep)
    df = sort_dataframe(df, ['Country', 'Year'])
    df = reset_dataframe_index(df)
    df['Air_pollution_level'] = pd.to_numeric(df['Air_pollution_level'].round(2), errors='coerce')
    df['Year'] = df['Year'].astype(int)
    return df

def process_development_index(filepath):
    """
    Procesa los datos del índice de desarrollo humano (IDH).
    
    Parámetros:
    filepath (str): La ruta del archivo Excel que contiene los datos del índice de desarrollo humano.
    
    Retorna:
    pd.DataFrame: El DataFrame procesado con datos del IDH, filtrado y ordenado.
    """
    sheet_name='Hoja 1'
    df = load_excel_data(filepath, sheet_name)
    df = df.rename(columns={'Unnamed: 1': 'Country', 'Unnamed: 2': 'IDH'})
    columnas_interes = ['Country', 'IDH']
    df = select_columns(df, columnas_interes)
    df = df.iloc[6:].reset_index(drop=True)
    df = filter_countries(df, countries_to_keep)
    df = sort_dataframe(df, ['Country'])
    df = reset_dataframe_index(df)
    return df

def fillna_with_mean(df, column):
    """
    Rellena los NaN en una columna con la media del valor anterior y posterior para el mismo país.
    Si solo uno de los valores (anterior o posterior) existe, se usa ese valor.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene la columna a modificar y las columnas 'Country' y 'Year'.
    column (str): El nombre de la columna en la que se rellenarán los NaN.
    
    Retorna:
    pd.DataFrame: El DataFrame con los NaN rellenados donde fue posible.
    """
    # Ordenar los datos por 'Country' y 'Year' para asegurar que están bien organizados.
    df = df.sort_values(by=['Country', 'Year']).reset_index(drop=True)
    
    # Iterar sobre las filas del DataFrame
    for i in range(len(df)):
        # Si el valor es NaN
        if pd.isna(df.loc[i, column]):
            country = df.loc[i, 'Country']
            year = df.loc[i, 'Year']

            # Obtener las filas anterior y posterior del mismo país
            prev_row = df[(df['Country'] == country) & (df['Year'] < year)].tail(1)
            next_row = df[(df['Country'] == country) & (df['Year'] > year)].head(1)
            
            # Obtener los valores anterior y posterior, si existen
            prev_value = prev_row[column].values[0] if not prev_row.empty else None
            next_value = next_row[column].values[0] if not next_row.empty else None
            
            # Rellenar con la media de los valores disponibles
            if pd.notna(prev_value) and pd.notna(next_value):
                df.loc[i, column] = (prev_value + next_value) / 2
            elif pd.notna(prev_value):  # Si solo hay valor anterior
                df.loc[i, column] = prev_value
            elif pd.notna(next_value):  # Si solo hay valor posterior
                df.loc[i, column] = next_value

    return df

def save_to_csv(df, file_path, index=False):
    """
    Guarda un DataFrame como archivo CSV.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame a guardar.
    file_path (str): La ruta y el nombre del archivo CSV de destino.
    index (bool): Si deseas incluir el índice en el CSV. Por defecto es False.
    
    Retorna:
    None: La función guarda el archivo en la ruta especificada.
    """
    try:
        df.to_csv(file_path, index=index)
        print(f"Archivo guardado correctamente en: {file_path}")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

def agregar_porcentajes(df, column):
    """
    Añade una columna al DataFrame que contiene el porcentaje de los valores en la columna especificada
    respecto al total de esa columna para cada año.

    Parámetros:
    df (pd.DataFrame): Un DataFrame que contiene al menos las columnas 'Year' y otra columna con valores numéricos.
    column (str): El nombre de la columna que contiene los valores numéricos para los que se calcularán los porcentajes.
    
    Retorna:
    pd.DataFrame: El DataFrame original con una nueva columna '<column>%' que muestra el porcentaje.
    """
    # Agrupar por año y calcular el total de la columna especificada para cada año
    total_por_año = df.groupby('Year')[column].transform('sum')
    
    # Calcular el porcentaje para cada fila y redondear a 2 decimales
    df[column + '%'] = ((df[column] / total_por_año) * 100).round(2)
    
    return df
