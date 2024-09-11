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
    """Carga datos desde un archivo CSV."""
    return pd.read_csv(filepath)

def load_excel_data(filepath, sheet_name):
    """Carga datos desde un archivo Excel."""
    return pd.read_excel(filepath, engine='openpyxl', sheet_name=sheet_name)

def filter_by_year(df, start_year, end_year):
    """Filtra el DataFrame por el rango de años dado."""
    return df[(df['TIME_PERIOD'] <= end_year) & (df['TIME_PERIOD'] >= start_year)]

def select_columns(df, columns):
    """Selecciona columnas específicas del DataFrame."""
    return df[columns]

def rename_columns(df, new_column_names):
    """Renombra las columnas del DataFrame."""
    df.columns = new_column_names
    return df

def replace_country_codes(df, country_dict):
    """Reemplaza los códigos de países con nombres completos."""
    df['Country'] = df['Country'].replace(country_dict)
    return df

def filter_countries(df, countries_to_keep):
    """Filtra el DataFrame para conservar solo los países listados."""
    return df[df['Country'].isin(countries_to_keep)]

def sort_dataframe(df, sort_columns):
    """Ordena el DataFrame por las columnas especificadas."""
    return df.sort_values(by=sort_columns)

def reset_dataframe_index(df):
    """Reinicia el índice del DataFrame."""
    return df.reset_index(drop=True)

def melt_dataframe(df, id_vars, var_name, value_name):
    """Convierte el DataFrame al formato largo."""
    return df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)

def process_recycling_data(filepath):
    """Procesa los datos de tasas de reciclaje."""
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
    """Procesa los datos de generación de residuos."""
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
    """Procesa los datos de emisiones de CO2."""
    
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
    """Procesa los datos de emisiones de CO2."""
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
    """Procesa los datos de emisiones de CO2."""
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

# Función para reemplazar NaN con la media del valor anterior y posterior
def fillna_with_mean(df, column):
    """Rellena los NaN con la media del valor anterior y posterior en una columna."""
    for i in range(1, len(df[column]) - 1):  # Evitar el primer y último índice
        if pd.isna(df.loc[i, column]):
            prev_value = df.loc[i - 1, column]
            next_value = df.loc[i + 1, column]
            if not pd.isna(prev_value) and not pd.isna(next_value):  # Ambos deben no ser NaN
                df.loc[i, column] = (prev_value + next_value) / 2
    return df
def save_to_csv(df, file_path, index=False):
    """
    Guarda un DataFrame como archivo CSV.
    
    Args:
    - df (pd.DataFrame): El DataFrame a guardar.
    - file_path (str): La ruta y el nombre del archivo CSV de destino.
    - index (bool): Si deseas incluir el índice en el CSV. Por defecto es False.
    
    Returns:
    - None: La función guarda el archivo en la ruta especificada.
    """
    try:
        df.to_csv(file_path, index=index)
        print(f"Archivo guardado correctamente en: {file_path}")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")