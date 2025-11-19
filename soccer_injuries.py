import random
from datetime import date, timedelta
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# ====================================================================
# 1. VARIABLES GLOBALES Y PARÁMETROS BASE
# ====================================================================

# Definición de Perfiles de Jugadores (11 Titulares + 7 Suplentes)
# Estructura: (Nombre Completo, Factor_Resistencia, Minutos_Media_Partido, Probabilidad_Jugar_Partido)
JUGADORES_PERFILES = [
    # 11 Titulares (Alta participación)
    ("Martín Silva", 1.2, 90, 0.98),  # Arquero - Muy Resistente
    ("Javier González", 1.1, 85, 0.95),  # Defensa - Resistente
    ("Andrés López", 1.0, 80, 0.90),  # Defensa - Promedio
    ("Ricardo Pérez", 0.9, 75, 0.88),  # Defensa - Baja Resistencia
    ("Carlos Romero", 1.0, 88, 0.92),  # Mediocampo - Box-to-Box
    ("Pablo Díaz", 1.05, 80, 0.91),  # Mediocampo - Promedio Alto
    ("Miguel Torres", 1.1, 70, 0.85),  # Mediocampo - Creador
    ("Sebastián Castro", 1.0, 80, 0.90),  # Mediocampo - Promedio
    ("Nicolás Vargas", 0.95, 90, 0.96),  # Delantero - Promedio Bajo
    ("Diego Herrera", 1.0, 85, 0.90),  # Delantero - Promedio
    ("Juan Méndez", 1.0, 80, 0.90),  # Lateral - Promedio
    # 7 Suplentes (Baja participación)
    ("Luis Soto", 0.9, 30, 0.60),  # Suplente - Joven
    ("Emilio Guzmán", 1.2, 45, 0.70),  # Suplente - Resistente
    ("Alejandro Ruiz", 0.8, 20, 0.50),  # Suplente - Frágil
    ("Esteban Ríos", 1.0, 35, 0.65),  # Suplente - Promedio
    ("Roberto Peña", 0.95, 40, 0.75),  # Suplente - Veterano
    ("Fernando Vidal", 1.1, 50, 0.80),  # Suplente - Utilidad
    ("Gabriel Rojas", 1.0, 5, 0.10),  # Suplente - Backup GK
]

# Configuración por defecto (modificable por el usuario)
CONFIG = {
    "PARTIDOS_TEMPORADA": 34,
    "FECHA_INICIO": date(2025, 11, 24), # Lunes
    "LAMBDA_BASE_MEDIA": 0.05,        # Riesgo base por semana
    "PESO_ENTRENAMIENTO": 0.005,      # Riesgo añadido por día de entrenamiento
    "PESO_MINUTOS_JUGADOS": 0.001,    # Riesgo añadido por minuto jugado
    "MIN_DIAS_BAJA": 1,               # Mínimo días de baja (entero)
    "MAX_DIAS_BAJA": 15               # Máximo días de baja (entero)
}

# ====================================================================
# 2. FUNCIONES DE CONFIGURACIÓN Y MENÚ
# ====================================================================

def mostrar_menu_configuracion():
    """Permite al usuario modificar los valores probabilísticos."""
    
    print("\n" + "=" * 50)
    print("##  CONFIGURACIÓN DE PARÁMETROS DE SIMULACIÓN")
    print("=" * 50)
    
    # 1. Lambda Base
    current_lambda = CONFIG["LAMBDA_BASE_MEDIA"]
    try:
        new_lambda = input(f"1. Lambda Base (Riesgo/Semana) [Actual: {current_lambda}]: ")
        if new_lambda:
            CONFIG["LAMBDA_BASE_MEDIA"] = float(new_lambda)
    except ValueError:
        print("Valor inválido. Se mantendrá el valor por defecto.")

    # 2. Peso Entrenamiento
    current_peso_e = CONFIG["PESO_ENTRENAMIENTO"]
    try:
        new_peso_e = input(f"2. Peso Entrenamiento (Riesgo/Día) [Actual: {current_peso_e}]: ")
        if new_peso_e:
            CONFIG["PESO_ENTRENAMIENTO"] = float(new_peso_e)
    except ValueError:
        print("Valor inválido. Se mantendrá el valor por defecto.")

    # 3. Peso Minutos Jugados
    current_peso_m = CONFIG["PESO_MINUTOS_JUGADOS"]
    try:
        new_peso_m = input(f"3. Peso Minutos (Riesgo/Minuto) [Actual: {current_peso_m}]: ")
        if new_peso_m:
            CONFIG["PESO_MINUTOS_JUGADOS"] = float(new_peso_m)
    except ValueError:
        print("Valor inválido. Se mantendrá el valor por defecto.")

    # 4. Días de Baja (Mínimo/Máximo)
    current_min_baja, current_max_baja = CONFIG["MIN_DIAS_BAJA"], CONFIG["MAX_DIAS_BAJA"]
    try:
        new_min_baja = input(f"4. Min. Días de Baja (Entero) [Actual: {current_min_baja}]: ")
        if new_min_baja:
            CONFIG["MIN_DIAS_BAJA"] = int(new_min_baja)
        new_max_baja = input(f"5. Max. Días de Baja (Entero) [Actual: {current_max_baja}]: ")
        if new_max_baja:
            CONFIG["MAX_DIAS_BAJA"] = int(new_max_baja)
    except ValueError:
        print("Valor inválido. Se mantendrá el valor por defecto.")
        
    print("\n Configuración guardada. Iniciando simulación...")
    print("-" * 50)

# ====================================================================
# 3. FUNCIONES DE SIMULACIÓN
# ====================================================================

def simular_dias_baja():
    """Simula los días de baja (entero) usando la configuración actual."""
    # Los días son enteros
    return random.randint(CONFIG["MIN_DIAS_BAJA"], CONFIG["MAX_DIAS_BAJA"])

def calcular_riesgo_lesion(factor_resistencia, carga_entrenamiento, minutos_jugados):
    """
    Calcula el parámetro de tasa (lambda) del modelo de riesgo.
    Aplica la Distribución Exponencial (lambda) para el análisis de riesgo.
    """
    
    # Riesgo = (Lambda Base / Resistencia) + Carga Entrenamiento + Carga Partido
    riesgo_lambda = (CONFIG["LAMBDA_BASE_MEDIA"] / factor_resistencia) + \
                    (carga_entrenamiento * CONFIG["PESO_ENTRENAMIENTO"]) + \
                    (minutos_jugados * CONFIG["PESO_MINUTOS_JUGADOS"])
    
    # P(Lesión en la semana) = 1 - e^(-lambda * T), con T=1 (semana)
    prob_lesion_semanal = 1 - math.exp(-riesgo_lambda) 
    return riesgo_lambda, prob_lesion_semanal

def simular_jugador(nombre, factor_resistencia, minutos_media_partido, prob_jugar_partido):
    """Simula una temporada completa para un jugador con sus parámetros específicos."""
    
    registro_temporada = []
    dias_activo_jugador = 0
    dias_baja_total = 0
    minutos_jugados_total = 0
    dias_baja_actual = 0
    fecha_actual = CONFIG["FECHA_INICIO"]
    
    lesiones_historial = []

    # Bucle principal de 34 semanas
    for semana in range(1, CONFIG["PARTIDOS_TEMPORADA"] + 1):
        
        carga_entrenamiento_semanal = 0
        minutos_jugados_semanales = 0
        
        log_semana = {
            "Semana": semana,
            "Fecha_Inicio_Semana": fecha_actual.strftime('%Y-%m-%d'),
            "Dias_Entrenamiento_Real": 0,
            "Minutos_Jugados_Partido": 0,
            "Lesion_Esta_Semana": False,
            "Dias_Baja_Inicio": 0,
            "Dia_Lesion": None
        }

        # Simulación diaria (Lunes a Domingo)
        for dia_indice in range(7):
            
            # --- 1. Lesión Activa (Recuperación) ---
            if dias_baja_actual > 0:
                dias_baja_actual -= 1
                fecha_actual += timedelta(days=1)
                continue

            # --- 2. Entrenamiento (Lunes a Viernes) ---
            if dia_indice < 5: # Lunes (0) a Viernes (4)
                if random.random() < 0.95: 
                    carga_entrenamiento_semanal += 1
                    dias_activo_jugador += 1
            
            # --- 3. Partido (Domingo) ---
            elif fecha_actual.weekday() == 6: # Domingo
                
                # Probabilidad de jugar basada en el perfil
                if random.random() < prob_jugar_partido: 
                    # Simulación de minutos (RANDOM)
                    minutos_jugados = max(0, min(90, round(random.normalvariate(minutos_media_partido, 10))))
                else:
                    minutos_jugados = 0
                
                minutos_jugados_semanales = minutos_jugados
                minutos_jugados_total += minutos_jugados
                if minutos_jugados > 0:
                    dias_activo_jugador += 1

                # --- 4. Evaluación de Riesgo y Lesión (Modelado de Eventos Discretos) ---
                riesgo_lambda, prob_lesion_semanal = calcular_riesgo_lesion(
                    factor_resistencia, carga_entrenamiento_semanal, minutos_jugados_semanales
                )
                
                # Lesión Ocurre (evento discreto)
                if random.random() < prob_lesion_semanal:
                    dias_baja_obtenidos = simular_dias_baja()
                    dias_baja_actual = dias_baja_obtenidos
                    dias_baja_total += dias_baja_obtenidos
                    
                    log_semana["Lesion_Esta_Semana"] = True
                    log_semana["Dias_Baja_Inicio"] = dias_baja_obtenidos
                    # Días de lesión son fechas reales
                    log_semana["Dia_Lesion"] = fecha_actual.strftime('%Y-%m-%d')
                    
                    lesiones_historial.append({
                        "Semana": semana,
                        "Fecha": log_semana["Dia_Lesion"],
                        "Dias_Baja": dias_baja_obtenidos,
                        "Riesgo_Lambda": round(riesgo_lambda, 4)
                    })
                    
            fecha_actual += timedelta(days=1)
        
        log_semana["Dias_Entrenamiento_Real"] = carga_entrenamiento_semanal
        log_semana["Minutos_Jugados_Partido"] = minutos_jugados_semanales
        registro_temporada.append(log_semana)

    # Resultados Finales
    dias_temporada = CONFIG["PARTIDOS_TEMPORADA"] * 7
    # Días de descanso son fechas reales (días sin entrenamiento, partido o lesión)
    dias_descanso = dias_temporada - dias_activo_jugador - dias_baja_total
    
    # Formato H:M:S para el reporte
    horas = minutos_jugados_total // 60
    minutos = minutos_jugados_total % 60
    tiempo_jugado_formato = f"{horas:02d}:{minutos:02d}:00" 

    resultados_finales = {
        "Jugador": nombre,
        "Factor_Resistencia": factor_resistencia,
        "Total_Lesiones": len(lesiones_historial),
        "Total_Dias_Baja": dias_baja_total,
        "Total_Minutos_Jugados_Int": minutos_jugados_total,
        "Total_Minutos_Formato": tiempo_jugado_formato,
        "Total_Dias_Entrenamiento": sum(log["Dias_Entrenamiento_Real"] for log in registro_temporada),
        "Total_Dias_Activos": dias_activo_jugador,
        "Total_Dias_Descanso": dias_descanso,
        "Historial_Lesiones": lesiones_historial
    }
    
    return resultados_finales, registro_temporada

# ====================================================================
# 4. FUNCIONES DE REPORTE Y VISUALIZACIÓN
# ====================================================================

def generar_reportes(resultados_jugadores, registros_semanales_todos):
    """Genera el archivo Excel y todas las imágenes de reportes."""
    
    # Crear la carpeta de reportes
    carpeta_reportes = "reportes"
    os.makedirs(carpeta_reportes, exist_ok=True)
    
    df_acumulado = pd.DataFrame(resultados_jugadores)
    
    # --- A. Preparación del DataFrame para reportes ---
    df_reporte_acumulado = df_acumulado[[
        'Jugador', 'Factor_Resistencia', 'Total_Lesiones', 'Total_Dias_Baja', 
        'Total_Minutos_Formato', 'Total_Dias_Entrenamiento', 'Total_Dias_Activos', 
        'Total_Dias_Descanso'
    ]].rename(columns={
        'Factor_Resistencia': 'Resistencia',
        'Total_Dias_Baja': 'Días Baja',
        'Total_Minutos_Formato': 'Min. Jug. (H:M:S)',
        'Total_Dias_Entrenamiento': 'Días Entr.',
        'Total_Dias_Activos': 'Días Act.',
        'Total_Dias_Descanso': 'Días Desc.'
    })
    
    # --- B. Generación de Imagen de Tabla Acumulada ---
    
    fig, ax = plt.subplots(figsize=(12, (len(df_reporte_acumulado) / 3) + 3)) 
    ax.axis('off')
    ax.axis('tight')
    
    # Crear tabla matplotlib (para imagen)
    tabla = ax.table(
        cellText=df_reporte_acumulado.values,
        colLabels=df_reporte_acumulado.columns,
        loc='center',
        cellLoc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.5)
    
    ax.set_title("Resultados Acumulados de la Temporada", fontsize=16)
    
    archivo_tabla = os.path.join(carpeta_reportes, "tabla_resultados_acumulados.png")
    plt.savefig(archivo_tabla)
    plt.close()
    print(f" Imagen de Tabla guardada en: **{archivo_tabla}**")
    
    # --- C. Generación de Gráficos (Imágenes sueltas) ---
    
    df_grafico = df_acumulado.rename(columns={
        'Total_Minutos_Jugados_Int': 'Min. Jug. Int', 
        'Total_Dias_Baja': 'Días Baja'
    }).sort_values(by='Días Baja', ascending=False)
    
    # Gráfico 1: Días de Baja vs. Minutos Jugados (Comparación)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(df_grafico['Jugador'], df_grafico['Días Baja'], color='skyblue', label='Días de Baja')
    ax1.set_ylabel('Total Días de Baja', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xlabel('Jugador')
    ax1.set_title('Predicción de Semanas de Baja y Minutos Jugados por Jugador')
    ax1.set_xticklabels(df_grafico['Jugador'], rotation=45, ha='right')
    
    ax2 = ax1.twinx() 
    ax2.plot(df_grafico['Jugador'], df_grafico['Min. Jug. Int'], color='red', marker='o', linestyle='--', label='Minutos Jugados')
    ax2.set_ylabel('Total Minutos Jugados', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.tight_layout()
    archivo_grafico_1 = os.path.join(carpeta_reportes, "grafico_dias_baja_vs_minutos.png")
    plt.savefig(archivo_grafico_1)
    plt.close()

    # Gráfico 2: Resistencia vs. Lesiones (Análisis de Riesgo)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_grafico['Factor_Resistencia'], df_grafico['Días Baja'], c=df_grafico['Factor_Resistencia'], cmap='RdYlGn', edgecolors='k')
    plt.title('Días de Baja vs. Factor de Resistencia Individual (Análisis de Riesgo)')
    plt.xlabel('Factor de Resistencia (Mayor = Menor Riesgo)')
    plt.ylabel('Total Días de Baja')
    plt.grid(True)
    archivo_grafico_2 = os.path.join(carpeta_reportes, "grafico_resistencia_vs_lesiones.png")
    plt.savefig(archivo_grafico_2)
    plt.close()
    print(f" Gráficos guardados en: **{carpeta_reportes}/**")

    # --- D. Exportar a Excel con hojas separadas por jugador ---
    nombre_archivo_excel = os.path.join(carpeta_reportes, "simulacion_detallada_resultados.xlsx")
    with pd.ExcelWriter(nombre_archivo_excel, engine='openpyxl') as writer:
        
        df_reporte_acumulado.to_excel(writer, sheet_name='01_Resultados_Acumulados', index=False)
        
        # Historial de Lesiones
        lesiones_flat = []
        for resultado in resultados_jugadores:
            for lesion in resultado['Historial_Lesiones']:
                lesiones_flat.append({
                    'Jugador': resultado['Jugador'],
                    'Resistencia': resultado['Factor_Resistencia'],
                    'Semana': lesion['Semana'],
                    'Fecha_Lesion': lesion['Fecha'],
                    'Dias_Baja': lesion['Dias_Baja'],
                    'Riesgo_Lambda': lesion['Riesgo_Lambda']
                })
        df_historial = pd.DataFrame(lesiones_flat)
        df_historial.to_excel(writer, sheet_name='02_Historial_Lesiones', index=False)
        
        # Pestañas individuales por jugador (CORRECCIÓN IMPLEMENTADA AQUÍ)
        # Usamos la lista plana de registros semanales directamente
        df_semanal_full = pd.DataFrame(registros_semanales_todos) 
        
        for jugador in df_semanal_full['Jugador'].unique():
            df_jugador = df_semanal_full[df_semanal_full['Jugador'] == jugador].drop(columns=['Jugador'])
            sheet_name = f"Detalle_{jugador}".replace(' ', '_')[:31] 
            df_jugador.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f" Archivo Excel guardado en: **{nombre_archivo_excel}**")

# ====================================================================
# 5. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ====================================================================

if __name__ == "__main__":
    
    # 1. Mostrar y aplicar configuración
    mostrar_menu_configuracion()
    
    # 2. Ejecutar Simulación
    resultados_jugadores = []
    registros_semanales_todos = []
    
    print("\n--- 🏃 Ejecutando Simulación de 34 Semanas ---")

    for i, (nombre, factor_r, min_med, prob_jugar) in enumerate(JUGADORES_PERFILES):
        sys.stdout.write(f"\rSimulando jugador {i+1}/18: {nombre}...")
        sys.stdout.flush()
        
        res_finales, reg_semanal = simular_jugador(
            nombre, factor_r, min_med, prob_jugar
        )
        
        for log in reg_semanal:
            # Asegurar que la clave 'Jugador' esté en cada diccionario
            log['Jugador'] = nombre
            registros_semanales_todos.append(log)
            
        resultados_jugadores.append(res_finales)

    # 3. Generar Reportes
    print("\n---  Simulación Completa. Generando Reportes... ---")
    generar_reportes(resultados_jugadores, registros_semanales_todos)
    
    print("\n--- Proyecto Terminado. Revisa la carpeta 'reportes' ---")