# 📦 Examen Final — Curso 709718
## Análisis de Tráfico Portuario Marítimo en Colombia

> **Curso:** 709718 — Herramientas computacionales para interpretación y validación de resultados  
> **Profesor:** Dr. Néstor Alzate Mejía  
> **Dataset:** `Trafico_Portuario_Marítimo_En_Colombia_20260511.csv`

---

## 🚀 Cómo ejecutar (desde cero)

### 1. Clonar el repositorio

```bash
git clone https://github.com/juanortizpa/examen-final-709718-portuario.git
cd examen-final-709718-portuario
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Agregar el dataset original

Copiar el archivo `Trafico_Portuario_Marítimo_En_Colombia_20260511.csv` en la carpeta `datos_originales/`:

```
examen-final-709718-portuario/
└── datos_originales/
    └── Trafico_Portuario_Marítimo_En_Colombia_20260511.csv   ← aquí
```

> ⚠️ El CSV no está en el repo por su tamaño. Descargarlo desde [Datos Abiertos Colombia](https://www.datos.gov.co/).

### 4. Ejecutar los scripts

```bash
python analisis_completo.py   # Limpieza, EDA, Figuras 1–3 y HHI
python analisis_parte2.py     # Modelos, validación, Figuras 4–5
```

---

## 📁 Estructura del proyecto

```
examen-final-709718-portuario/
├── analisis_completo.py          # Script 1: limpieza, EDA, figuras 1-3
├── analisis_parte2.py            # Script 2: modelos, validación, figuras 4-5
├── requirements.txt              # Dependencias Python
├── datos_originales/             # ← colocar el CSV aquí (no incluido en repo)
├── datos_procesados/
│   └── datos_limpios.csv         # Dataset tras limpieza (generado por script 1)
├── figuras/
│   ├── figura_1_calidad_datos.png
│   ├── figura_2_serie_temporal.png
│   ├── figura_3_comparacion_grupos.png
│   ├── figura_4_observado_predicho.png
│   ├── figura_5_error_residuos.png
│   └── figura_hallazgo1_HHI.png
└── resultados/
    ├── metricas_modelos.csv
    ├── errores_por_segmento.csv
    └── concentracion_HHI.csv
```

---

## 📊 Resultados principales

| Modelo | RMSE Validación | nRMSE | R² Val |
|--------|----------------|-------|--------|
| M0 — Línea base (promedio zona-mes) | 305.822 t | 23.1% | 0.943 |
| M1 — Regresión Lineal | 984.643 t | 74.3% | 0.404 |
| **M2A — RF n=50, prof=5** ✅ | **311.686 t** | **23.5%** | **0.940** |
| M2B — RF n=100, prof=8 | 313.952 t | 23.7% | 0.939 |
| M2C — RF n=100, sin límite | 324.328 t | 24.5% | 0.935 |

> Validación temporal holdout: **entrenamiento 2018–2023 / validación 2024–2025**

---

## 🔍 Hallazgos principales

1. **Concentración portuaria (HHI 1.598–1.776):** el Top 3 de zonas concentra el 60.8% del tráfico nacional. La concentración aumentó en 2020 (COVID) y se ha mantenido moderada-alta.

2. **Paradoja RMSE absoluto vs relativo:** zonas con RMSE absoluto bajo (TUMACO: 37.713 t) tienen nRMSE extremo (2.202%). El modelo global es inutilizable para zonas de bajo volumen.

3. **Fuga temporal en k-fold:** la validación k-fold aleatoria subestima el error real en **19–27%** frente a la validación temporal. Usar k-fold genera falsa confianza en el modelo.

---

## 🛠 Dependencias

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

---

## 🤖 Uso de IA

El análisis fue apoyado con Claude (Anthropic) para estructura del código, redacción del informe e interpretación de resultados. Todos los resultados numéricos fueron verificados ejecutando el código sobre el dataset real. Ver sección 9 del informe final para la bitácora completa.

---

*Universidad Cooperativa de Colombia — Facultad de Ingeniería*
