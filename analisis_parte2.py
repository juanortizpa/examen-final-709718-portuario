"""
EXAMEN FINAL — Curso 709718  |  Parte 2: Modelado, validación y hallazgos
"""
import os, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings('ignore')

# ── RUTAS RELATIVAS ─────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
PROC    = os.path.join(BASE, "datos_procesados")
FIGS    = os.path.join(BASE, "figuras")
RESULTS = os.path.join(BASE, "resultados")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

df = pd.read_csv(os.path.join(PROC, "datos_limpios.csv"), parse_dates=['FECHA'])

FLUJOS = ['EXPORTACION','IMPORTACIÓN','TRANSBORDO','TRANSITO INTERNACIONAL',
          'FLUVIAL','CABOTAJE','MOVILIZACIONES A BORDO','TRANSITORIA']

df['TRIMESTRE'] = ((df['MES VIGENCIA']-1)//3)+1
zm = df.groupby(['ZONA PORTUARIA','AÑO VIGENCIA','MES VIGENCIA','PERIODO_NUM','FECHA','TRIMESTRE']).agg(
    TOTAL=('TOTAL_MOVILIZADO','sum'),
    EXPORTACION=('EXPORTACION','sum'),
    IMPORTACION=('IMPORTACIÓN','sum'),
    N_REGISTROS=('TOTAL_MOVILIZADO','count')
).reset_index()

zm_model = zm[zm['AÑO VIGENCIA'] < 2026].copy().sort_values('PERIODO_NUM').reset_index(drop=True)
le = LabelEncoder()
zm_model['ZONA_ENC']   = le.fit_transform(zm_model['ZONA PORTUARIA'])
zm_model['PROM_BASE']  = zm_model.groupby(['ZONA PORTUARIA','MES VIGENCIA'])['TOTAL'].transform('mean')

FEATURES = ['ZONA_ENC','AÑO VIGENCIA','MES VIGENCIA','PERIODO_NUM','TRIMESTRE','N_REGISTROS']
X = zm_model[FEATURES].values
y = zm_model['TOTAL'].values

train_mask = zm_model['AÑO VIGENCIA'] <= 2023
val_mask   = zm_model['AÑO VIGENCIA'] >= 2024
X_tr, y_tr = X[train_mask], y[train_mask]
X_va, y_va = X[val_mask],   y[val_mask]

print(f"Train (2018-2023): {train_mask.sum()} | Val (2024-2025): {val_mask.sum()}")

def metricas(y_true, y_pred, nombre, tag=''):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true-y_pred)/(y_true+1)))*100
    r2   = r2_score(y_true, y_pred)
    nrmse= rmse/(y_true.mean()+1)*100
    print(f"  {nombre}: MAE={mae:>10,.0f}  RMSE={rmse:>10,.0f}  MAPE={mape:>5.1f}%  nRMSE={nrmse:>5.1f}%  R²={r2:.3f}")
    return {'Modelo':nombre,'Set':tag,'MAE':mae,'RMSE':rmse,'MAPE':mape,'nRMSE':nrmse,'R2':r2}

resultados = []

print("\n=== MODELO 0: Línea base ===")
resultados.append(metricas(y_va, zm_model.loc[val_mask,'PROM_BASE'].values, "M0 - Promedio zona-mes", "VAL_TEMPORAL"))

print("\n=== MODELO 1: Regresión Lineal ===")
lr = LinearRegression().fit(X_tr, y_tr)
resultados.append(metricas(y_tr, lr.predict(X_tr), "M1 - Reg. Lineal", "TRAIN"))
resultados.append(metricas(y_va, lr.predict(X_va), "M1 - Reg. Lineal", "VAL_TEMPORAL"))

print("\n=== MODELO 2: Random Forest (3 configuraciones) ===")
configs = [
    (50,  5,    "M2A - RF n=50  prof=5"),
    (100, 8,    "M2B - RF n=100 prof=8"),
    (100, None, "M2C - RF n=100 sin límite"),
]
best_rmse, best_pred_va = np.inf, None
for n_est, depth, name in configs:
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p_tr = rf.predict(X_tr);  p_va = rf.predict(X_va)
    resultados.append(metricas(y_tr, p_tr, name, "TRAIN"))
    rv = metricas(y_va, p_va, name, "VAL_TEMPORAL")
    resultados.append(rv)
    if rv['RMSE'] < best_rmse:
        best_rmse, best_pred_va, best_name = rv['RMSE'], p_va, name

print(f"\n>>> Mejor modelo: {best_name} (RMSE val={best_rmse:,.0f})")

# K-Fold vs temporal
print("\n=== K-Fold Aleatorio vs Validación Temporal ===")
rf_kf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
kf_maes, kf_rmses = [], []
for tr_i, va_i in KFold(n_splits=5, shuffle=True, random_state=42).split(X, y):
    rf_kf.fit(X[tr_i], y[tr_i]); p = rf_kf.predict(X[va_i])
    kf_maes.append(mean_absolute_error(y[va_i], p))
    kf_rmses.append(np.sqrt(mean_squared_error(y[va_i], p)))
kf_mae, kf_rmse = np.mean(kf_maes), np.mean(kf_rmses)
m2b_val = [r for r in resultados if 'M2B' in r['Modelo'] and r['Set']=='VAL_TEMPORAL'][0]
print(f"  K-Fold aleatorio: MAE={kf_mae:>10,.0f}  RMSE={kf_rmse:>10,.0f}")
print(f"  Temporal        : MAE={m2b_val['MAE']:>10,.0f}  RMSE={m2b_val['RMSE']:>10,.0f}")
print(f"  K-Fold es {(kf_rmse-m2b_val['RMSE'])/m2b_val['RMSE']*100:+.1f}% en RMSE vs temporal")

# Error por zona
zm_val = zm_model[val_mask].copy()
zm_val['PRED'] = best_pred_va
zm_val['ERR_ABS'] = np.abs(zm_val['TOTAL'] - zm_val['PRED'])
zm_val['ERR_REL'] = zm_val['ERR_ABS'] / (zm_val['TOTAL']+1) * 100
err_zona = zm_val.groupby('ZONA PORTUARIA').apply(lambda g: pd.Series({
    'MAE':  mean_absolute_error(g['TOTAL'], g['PRED']),
    'RMSE': np.sqrt(mean_squared_error(g['TOTAL'], g['PRED'])),
    'ERROR_REL': g['ERR_REL'].mean(),
    'TOTAL_MEDIO': g['TOTAL'].mean()
})).reset_index()
err_zona['nRMSE'] = err_zona['RMSE'] / (err_zona['TOTAL_MEDIO']+1)*100

pd.DataFrame(resultados).to_csv(os.path.join(RESULTS, "metricas_modelos.csv"), index=False)
err_zona.to_csv(os.path.join(RESULTS, "errores_por_segmento.csv"), index=False)
print("\nMétricas guardadas en resultados/")

# ── FIGURA 4 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Figura 4. Observado vs. Predicho — Mejor RF (2024–2025)", fontsize=13, fontweight='bold')
max_v = max(y_va.max(), best_pred_va.max())/1e6
axes[0].scatter(y_va/1e6, best_pred_va/1e6, alpha=0.5, color='#2980b9', s=40, edgecolors='k', lw=0.2)
axes[0].plot([0,max_v],[0,max_v],'r--',lw=1.5,label='Ajuste perfecto')
axes[0].set_xlabel("Observado (M t)"); axes[0].set_ylabel("Predicho (M t)")
axes[0].set_title(f"RMSE={best_rmse/1e6:.2f}M t  R²={r2_score(y_va,best_pred_va):.3f}")
axes[0].legend()
fechas_va = zm_model.loc[val_mask,'FECHA']
df_t = pd.DataFrame({'FECHA':fechas_va.values,'OBS':y_va,'PRED':best_pred_va})
df_agg = df_t.groupby('FECHA')[['OBS','PRED']].sum().reset_index()
axes[1].plot(df_agg['FECHA'],df_agg['OBS']/1e6,'o-',color='#2c3e50',label='Observado')
axes[1].plot(df_agg['FECHA'],df_agg['PRED']/1e6,'s--',color='#e74c3c',label='Predicho')
axes[1].set_title("Serie mensual nacional — validación temporal")
axes[1].set_xlabel("Fecha"); axes[1].set_ylabel("Total (M t)"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "figura_4_observado_predicho.png"), dpi=150, bbox_inches='tight')
plt.close(); print("Figura 4 guardada.")

# ── FIGURA 5 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Figura 5. Análisis de errores — dónde falla el modelo", fontsize=13, fontweight='bold')
err_plot = err_zona.sort_values('ERROR_REL', ascending=True)
colors_e = ['#e74c3c' if v>50 else '#f39c12' if v>25 else '#27ae60' for v in err_plot['ERROR_REL']]
axes[0].barh(err_plot['ZONA PORTUARIA'], err_plot['ERROR_REL'], color=colors_e)
axes[0].axvline(25, color='orange', ls='--', lw=1.2, label='25%')
axes[0].axvline(50, color='red',    ls='--', lw=1.2, label='50%')
axes[0].set_title("Error relativo medio (%) por zona — 2024–2025")
axes[0].set_xlabel("Error relativo medio (%)"); axes[0].legend(fontsize=9)
zm_vs = zm_val.sort_values('FECHA')
residuos = zm_vs['TOTAL'] - zm_vs['PRED']
axes[1].scatter(zm_vs['FECHA'], residuos/1e6, alpha=0.4, color='#8e44ad', s=20)
axes[1].axhline(0, color='black', lw=1)
axes[1].set_title("Residuos en el tiempo — validación")
axes[1].set_xlabel("Fecha"); axes[1].set_ylabel("Residuo (M t)")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "figura_5_error_residuos.png"), dpi=150, bbox_inches='tight')
plt.close(); print("Figura 5 guardada.")

print("\n✓ analisis_parte2.py finalizado. Revisa las carpetas figuras/ y resultados/")
