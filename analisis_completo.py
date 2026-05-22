"""
EXAMEN FINAL — Curso 709718
Herramientas computacionales para interpretación y validación de resultados
Caso: Tráfico Portuario Marítimo en Colombia
"""
import os, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')

# ── RUTAS RELATIVAS (funciona en Windows, Mac y Linux) ──────────
BASE    = os.path.dirname(os.path.abspath(__file__))
RAW     = os.path.join(BASE, "datos_originales", "Trafico_Portuario_Marítimo_En_Colombia_20260511.csv")
PROC    = os.path.join(BASE, "datos_procesados")
FIGS    = os.path.join(BASE, "figuras")
RESULTS = os.path.join(BASE, "resultados")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

# ═══════════════════════════════════════════════════════════════
# 1. CARGA Y LIMPIEZA
# ═══════════════════════════════════════════════════════════════
df_raw = pd.read_csv(RAW)
df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]

num_cols = ['EXPORTACION','IMPORTACIÓN','TRANSBORDO','TRANSITO INTERNACIONAL',
            'FLUVIAL','CABOTAJE','MOVILIZACIONES A BORDO','TRANSITORIA','AÑO VIGENCIA']
for col in num_cols:
    df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['AÑO VIGENCIA']     = df['AÑO VIGENCIA'].astype(int)
df['MES VIGENCIA']     = df['MES VIGENCIA'].astype(int)
df['ZONA PORTUARIA']   = df['ZONA PORTUARIA'].str.strip()
df['SOCIEDAD PORTUARIA'] = df['SOCIEDAD PORTUARIA'].str.strip()
df['TIPO DE CARGA']    = df['TIPO DE CARGA'].str.strip()
df['TIPO DE SERVICIO'] = df['TIPO DE SERVICIO'].str.strip()
df = df.drop_duplicates().reset_index(drop=True)

print("="*55)
print("RESUMEN DE LIMPIEZA")
print("="*55)
print(f"Registros: {len(df)} | Años: {sorted(df['AÑO VIGENCIA'].unique())}")
print(f"Registros 2026 (incompleto): {(df['AÑO VIGENCIA']==2026).sum()}")
print(f"TRANSITORIA con cero: {(df['TRANSITORIA']==0).mean():.1%}")

# ═══════════════════════════════════════════════════════════════
# 2. VARIABLES DERIVADAS
# ═══════════════════════════════════════════════════════════════
FLUJOS = ['EXPORTACION','IMPORTACIÓN','TRANSBORDO','TRANSITO INTERNACIONAL',
          'FLUVIAL','CABOTAJE','MOVILIZACIONES A BORDO','TRANSITORIA']
df['TOTAL_MOVILIZADO'] = df[FLUJOS].sum(axis=1)
df['FECHA'] = pd.to_datetime(
    df['AÑO VIGENCIA'].astype(str)+'-'+df['MES VIGENCIA'].astype(str).str.zfill(2)+'-01')
df['PERIODO_NUM']    = (df['AÑO VIGENCIA']-2018)*12 + (df['MES VIGENCIA']-1)
df['BALANCE_EXP_IMP']= df['EXPORTACION'] - df['IMPORTACIÓN']
df['TRIMESTRE']      = ((df['MES VIGENCIA']-1)//3)+1
df['AÑO_INCOMPLETO'] = (df['AÑO VIGENCIA']==2026).astype(int)

df.to_csv(os.path.join(PROC, "datos_limpios.csv"), index=False)
print("\nDatos procesados guardados en datos_procesados/datos_limpios.csv")

# ═══════════════════════════════════════════════════════════════
# 3. FIGURA 1: COBERTURA Y CALIDAD
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 1. Cobertura y calidad del dataset\nTráfico Portuario Marítimo — Colombia 2018–2026",
             fontsize=13, fontweight='bold')
reg_año = df.groupby('AÑO VIGENCIA').size()
colors  = ['#e74c3c' if y==2026 else '#2980b9' for y in reg_año.index]
bars = axes[0].bar(reg_año.index, reg_año.values, color=colors, edgecolor='white', width=0.6)
axes[0].set_title("Registros por año (rojo = 2026 incompleto)")
axes[0].set_xlabel("Año"); axes[0].set_ylabel("N° registros")
for bar, val in zip(bars, reg_año.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(val),
                 ha='center', va='bottom', fontsize=9)
axes[0].set_ylim(0, reg_año.max()*1.15)
reg_zona = df.groupby('ZONA PORTUARIA').size().sort_values(ascending=True)
axes[1].barh(reg_zona.index, reg_zona.values, color='#27ae60', edgecolor='white')
axes[1].set_title("Registros por zona portuaria")
axes[1].set_xlabel("N° registros")
for i, v in enumerate(reg_zona.values):
    axes[1].text(v+3, i, str(v), va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "figura_1_calidad_datos.png"), dpi=150, bbox_inches='tight')
plt.close(); print("Figura 1 guardada.")

# ═══════════════════════════════════════════════════════════════
# 4. FIGURA 2: EVOLUCIÓN TEMPORAL
# ═══════════════════════════════════════════════════════════════
serie   = df.groupby('FECHA')['TOTAL_MOVILIZADO'].sum().reset_index()
mask26  = serie['FECHA'].dt.year == 2026
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(serie.loc[~mask26,'FECHA'], serie.loc[~mask26,'TOTAL_MOVILIZADO']/1e6,
        color='#2c3e50', lw=1.8, marker='o', ms=3, label='2018–2025')
ax.plot(serie.loc[mask26,'FECHA'], serie.loc[mask26,'TOTAL_MOVILIZADO']/1e6,
        color='#e74c3c', lw=2, marker='D', ms=5, ls='--', label='2026 (parcial)')
ax.annotate("Efecto COVID-19", xy=(pd.Timestamp('2020-05-01'), 30),
            xytext=(pd.Timestamp('2019-01-01'), 20),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
ax.set_title("Figura 2. Evolución temporal del total movilizado mensual — Colombia",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Fecha"); ax.set_ylabel("Toneladas (millones)"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "figura_2_serie_temporal.png"), dpi=150, bbox_inches='tight')
plt.close(); print("Figura 2 guardada.")

# ═══════════════════════════════════════════════════════════════
# 5. FIGURA 3: COMPARACIÓN POR GRUPOS
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Figura 3. Concentración y composición del tráfico portuario (2018–2025)",
             fontsize=13, fontweight='bold')
zona_total = df[df['AÑO VIGENCIA']<2026].groupby('ZONA PORTUARIA')['TOTAL_MOVILIZADO'].sum().sort_values(ascending=False)
pct = zona_total / zona_total.sum() * 100
cumsum = pct.cumsum()
colors_bar = ['#e74c3c','#e67e22','#f1c40f'] + ['#bdc3c7']*(len(pct)-3)
x = range(len(pct))
axes[0].bar(x, pct.values, color=colors_bar, edgecolor='white')
ax2 = axes[0].twinx()
ax2.plot(x, cumsum.values, 'k--o', ms=4, label='% Acumulado')
ax2.axhline(80, color='navy', ls=':', lw=1.2, label='80%')
ax2.set_ylim(0, 115); ax2.set_ylabel("% Acumulado")
axes[0].set_xticks(x); axes[0].set_xticklabels(pct.index, rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel("Participación (%)"); axes[0].set_xlabel("Zona portuaria")
axes[0].set_title(f"Participación por zona (Top 3 = {pct.head(3).sum():.1f}%)")
ax2.legend(loc='center right', fontsize=8)
flujo_total = df[df['AÑO VIGENCIA']<2026][FLUJOS].sum().sort_values(ascending=True)
pct_f = flujo_total/flujo_total.sum()*100
axes[1].barh(range(len(pct_f)), pct_f.values, color=sns.color_palette('viridis', len(pct_f)))
axes[1].set_yticks(range(len(pct_f))); axes[1].set_yticklabels(pct_f.index, fontsize=9)
for i, v in enumerate(pct_f.values):
    axes[1].text(v+0.3, i, f"{v:.1f}%", va='center', fontsize=9)
axes[1].set_title("Composición de flujos logísticos (2018–2025)")
axes[1].set_xlabel("Participación (%)")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "figura_3_comparacion_grupos.png"), dpi=150, bbox_inches='tight')
plt.close(); print("Figura 3 guardada.")

# ═══════════════════════════════════════════════════════════════
# 6. HHI — Hallazgo 1
# ═══════════════════════════════════════════════════════════════
def hhi_fn(s): sh = s/s.sum(); return (sh**2).sum()*10000
hhi_data  = df[df['AÑO VIGENCIA']<2026].groupby(['AÑO VIGENCIA','ZONA PORTUARIA'])['TOTAL_MOVILIZADO'].sum()
hhi_anual = hhi_data.groupby('AÑO VIGENCIA').apply(hhi_fn)
hhi_anual.to_csv(os.path.join(RESULTS, "concentracion_HHI.csv"))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hhi_anual.index, hhi_anual.values, 'o-', color='#8e44ad', lw=2, ms=8)
ax.axhspan(1500, 2500, alpha=0.1, color='orange', label='Concentración moderada-alta')
for y_val, v in zip(hhi_anual.index, hhi_anual.values):
    ax.annotate(f"{v:.0f}", (y_val, v+15), ha='center', fontsize=9)
ax.set_title("HHI de concentración portuaria — Colombia 2018–2025", fontsize=11, fontweight='bold')
ax.set_xlabel("Año"); ax.set_ylabel("HHI (0–10 000)")
ax.set_ylim(0, 2800); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "figura_hallazgo1_HHI.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figura HHI guardada.")
print("\n✓ analisis_completo.py finalizado. Ahora ejecuta analisis_parte2.py")
