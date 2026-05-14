# NFL Draft Prediction — Implementation Log

## Competencia
- **Evento**: GCI World 2026 April
- **Cierre**: 12 junio 2026, 11AM UTC
- **Métrica**: AUC (Area Under ROC Curve) — maximizar
- **Notebook de referencia**: `baseline.ipynb`
- **Submissions**: `results/submission_<fase>_<timestamp>.csv`

---

## Resultados por Fase

| Fase | Notebook | OOF AUC | Descripción |
|------|----------|----------|-------------|
| Baseline | `baseline.ipynb` | — | 0.80792 (public LB) | RandomForest, media global, sin flags |
| Fase 1 | `phase1_lgbm.ipynb` | 0.82056 | 0.81496 (public LB) | LightGBM + imputación grupal + missing flags |
| Fase 2 | `phase2_features.ipynb` | 0.82480 | 0.82174 (public LB) | + TE school/position + z-scores + overall_athleticism |
| ~~Fase 3~~ | `phase3_ensemble.ipynb` | 0.85130 | **0.81833 — DESCARTADA** (overfit severo) | Optuna HPO + Ensemble LGBM+XGB+RF |
| Fase 2b | `phase2b_interactions.ipynb` | 0.83830 | 0.81971 (public LB) | Base Fase 2 + feature interactions |

---

## Fase 1 — LightGBM + Group Imputation + Missing Flags
**Notebook**: `phase1_lgbm.ipynb`
**Submission**: `results/submission_phase1_2026-05-14_18-06-14.csv`
**OOF AUC**: 0.82056 (folds: 0.807 / 0.840 / 0.840 / 0.796 / 0.834)

### Cambios vs Baseline

#### 1. Modelo: RandomForest → LightGBM
- Gradient boosting supera a RF en datos tabulares
- LightGBM maneja NaN nativamente, pero aún imputamos para consistencia
- Parámetros: `lr=0.05`, `num_leaves=63`, `feature_fraction=0.8`, early stopping a 50 rondas
- Best iter por fold: 47 / 55 / 26 / 34 / 38 (modelos relativamente simples)

#### 2. Missing Flags
Columnas binarias `missing_<col>` por cada feature con NaN.

Señal observada (diff = ausente − presente):
| Feature | Presente | Ausente | Diff |
|---------|----------|---------|------|
| Age | 0.765 | 0.018 | **-0.747** (muy fuerte) |
| Sprint_40yd | 0.652 | 0.586 | -0.066 |
| Vertical_Jump | 0.664 | 0.585 | -0.079 |
| Bench_Press_Reps | 0.665 | 0.602 | -0.063 |
| Broad_Jump | 0.663 | 0.594 | -0.069 |
| Agility_3cone | 0.671 | 0.605 | -0.066 |
| Shuttle | 0.671 | 0.601 | -0.071 |

`missing_Age` resultó ser el feature más importante del modelo (gain: 5680).

#### 3. Imputación por Grupo (Position_Type)
Mediana por `Position_Type` en lugar de media global. Ejemplo:
- `Sprint_40yd`: backs_receivers 4.58s vs offensive_lineman 5.24s (diff de 0.66s)
- `Bench_Press_Reps`: defensive_lineman 26 reps vs defensive_back 15 reps

#### 4. Features usadas (21 total)
`Year, Age, Height, Weight, Sprint_40yd, Vertical_Jump, Bench_Press_Reps, Broad_Jump, Agility_3cone, Shuttle, Player_Type, Position_Type, Position, missing_Age, missing_Sprint_40yd, missing_Vertical_Jump, missing_Bench_Press_Reps, missing_Broad_Jump, missing_Agility_3cone, missing_Shuttle, BMI`

### Top Features por Gain
1. `missing_Age` — 5680 (dominante)
2. `Sprint_40yd` — 1282
3. `Weight` — 936
4. `BMI` — 840
5. `Bench_Press_Reps` — 559

### Notas y Observaciones
- `School` (236 únicos) sigue descartada — pendiente target encoding en Fase 2
- Las categóricas usan LabelEncoder — pendiente target encoding en Fase 2
- El gap entre folds (0.796 vs 0.840) sugiere que aún hay varianza; más datos o más features pueden estabilizarlo

---

---

## Fase 2 — Target Encoding + Z-scores + Overall Athleticism
**Notebook**: `phase2_features.ipynb`
**Submission**: `results/submission_phase2_2026-05-14_18-15-26.csv`
**OOF AUC**: 0.82480 (folds: 0.826 / 0.869 / 0.855 / 0.813 / 0.851)

### Cambios vs Fase 1

#### 1. Target Encoding (static, alpha=10)
- `School`, `Position`, `Position_Type` → `te_School`, `te_Position`, `te_Position_Type`
- TE estático calculado sobre todo el train (no fold-specific) con smoothing alpha=10
- Smoothing: `te = (n × mean_grupo + 10 × mean_global) / (n + 10)`
- `te_School` resultó el 3er feature más importante (gain: 890)
- Rango útil: Texas-El Paso 0.468 (baja tasa) → Northwestern 0.760 (alta tasa)
- **Nota**: TE fold-specific fue descartado — causaba inestabilidad severa (Fold 3 paraba en 4-6 iteraciones) por cambios drásticos en los valores para las 65 escuelas con 1 solo jugador

#### 2. Z-scores por Position (20 posiciones)
- Para cada PERF_COL: `z = (x - mean_position) / std_position`
- Tiempos (Sprint, Agility, Shuttle) invertidos (`z = -z`) para consistencia de dirección
- Stats calculadas solo en train, aplicadas a train y test
- `z_Sprint_40yd` es el 2do feature más importante (gain: 1212)

#### 3. Overall Athleticism
- Promedio de los 6 z-scores: resumen compacto del perfil atlético
- Gain: 585 (4to más importante)
- Las composites específicas (explosive_score, agility_composite, power_score) fueron eliminadas por ser combinaciones lineales de los z-scores — redundantes

### Features usadas (29 total)
Todas las de Fase 1 + `z_Sprint_40yd`, `z_Vertical_Jump`, `z_Bench_Press_Reps`, `z_Broad_Jump`, `z_Agility_3cone`, `z_Shuttle`, `overall_athleticism`, `te_School`, `te_Position`, `te_Position_Type`

### Lecciones aprendidas
- TE fold-specific: teóricamente correcto pero produce modelos inestables en datasets pequeños con muchos grupos raros
- TE estático con smoothing: práctica estándar en competencias, aceptable con alpha suficientemente grande
- Redundancia raw + z-scores: LGBM maneja ambos bien con regularización; los folds dejan de converger prematuramente

---

## Fase 3 — Optuna HPO + Ensemble
**Notebook**: `phase3_ensemble.ipynb`
**Submission**: `results/submission_phase3_2026-05-14_18-21-52.csv`
**OOF AUC**: 0.85130 (ensemble) — salto de +0.0265 vs Fase 2

### Resultados por modelo

| Modelo | OOF AUC | Peso ensemble | Optuna mejores params |
|--------|---------|---------------|----------------------|
| **LGBM** | 0.85112 | 0.3340 | lr=0.048, num_leaves=82, max_depth=5, min_child=75 |
| **XGB** | 0.85191 | 0.3343 | lr=0.094, max_depth=4, min_child_weight=8, gamma=4.03 |
| RF | 0.84530 | 0.3317 | 500 trees, min_samples_leaf=5, class_weight=balanced |

### Observaciones
- Optuna encontró que `num_leaves=82` y `max_depth=5` son más conservadores que los defaults — previene overfitting en este dataset pequeño
- `min_child_samples=75` en LGBM (vs default 20) — hoja mínima grande, fuerte regularización
- XGB tuvo `gamma=4.03` alto — fuerte regularización por pruning de nodos
- Los 3 modelos tienen pesos casi idénticos (~0.333) porque sus OOF AUC son muy similares
- Ensemble no ganó sobre el mejor modelo individual (-0.00061 OOF) — los modelos son correlacionados; el valor real del ensemble es robustez en test

### Hiperparámetros Optuna — LightGBM
```
learning_rate: 0.04793 | num_leaves: 82 | max_depth: 5
min_child_samples: 75  | feature_fraction: 0.776 | bagging_fraction: 0.766
reg_alpha: 0.01415     | reg_lambda: 0.03149     | min_split_gain: 0.07536
```

### Hiperparámetros Optuna — XGBoost
```
learning_rate: 0.09367 | max_depth: 4 | min_child_weight: 8
subsample: 0.5975      | colsample_bytree: 0.8392
reg_alpha: 0.11945     | reg_lambda: 8.6825 | gamma: 4.0309
```

---

## Fase 2b — Feature Interactions
**Notebook**: `phase2b_interactions.ipynb`
**Submission**: `results/submission_phase2b_2026-05-14_18-36-48.csv`
**OOF AUC**: 0.83830 (+0.01350 vs Fase 2)

### Features nuevas (17) — todas contribuyen con gain > 0

| Categoría | Feature | Gain | Correlación con Drafted |
|-----------|---------|------|------------------------|
| Z-score interaction | `z_sprint_x_bench` | 354 | — (position-normalized) |
| Ratio drill | `speed_agility_ratio` | 353 | -0.115 |
| Ratio físico | `strength_per_weight` | 292 | +0.092 |
| Ratio drill | `agility_shuttle_ratio` | 269 | +0.031 |
| Producto físico | `power_speed` (Weight/Sprint) | 254 | +0.146 |
| Z-score interaction | `z_sprint_x_vertical` | 240 | — |
| Z-score interaction | `z_sprint_x_broad` | 236 | — |
| Ratio físico | `jump_reach_ratio` | 214 | +0.075 |
| Producto rendimiento | `sprint_x_bench` | 199 | +0.153 |
| Ratio físico | `broad_per_height` | 191 | +0.058 |
| Ratio físico | `weight_per_inch` | 184 | +0.070 |

### Observaciones
- 7 de las 17 nuevas features entran en el Top 15 global
- Las interacciones de z-scores (`z_sprint_x_bench`, etc.) son más informativas que las de los valores raw (`sprint_x_bench`), porque están position-normalized — LGBM puede generalizar mejor entre posiciones
- `speed_agility_ratio` y `agility_shuttle_ratio` capturan perfiles de agilidad que las métricas individuales no capturaban
- `strength_per_weight` (fuerza relativa al peso) es conceptualmente similar al BMI pero para rendimiento
- `sprint_sq` tuvo el menor gain (49) — la no-linealidad de sprint ya estaba capturada por los otros features

---

## Fase 2c — Selección quirúrgica + Regularización fuerte
**Notebook**: `phase2c_selective.ipynb`
**Submission**: `results/submission_phase2c_2026-05-14_18-44-47.csv`
**OOF AUC**: 0.83209 (+0.007 vs Fase 2 | más moderado que 2b +0.013)

### Cambios vs Fase 2

| # | Feature nueva | Tipo | Gain esperado |
|---|---------------|------|---------------|
| 1 | `n_tests_completed` | Resumen pre-imputación (0-6 tests) | informativa |
| 2 | `z_sprint_x_bench` | Z-score interaction (velocidad × fuerza) | top de Fase 2b |
| 3 | `speed_agility_ratio` | Sprint/Agility (perfil velocidad vs agilidad) | top de Fase 2b |
| 4 | `strength_per_weight` | Fuerza relativa al peso | top de Fase 2b |
| 5 | `agility_shuttle_ratio` | Relación entre los dos drills de agilidad | top de Fase 2b |
| 6 | `power_speed` | Peso/Sprint (inercia en velocidad) | top de Fase 2b |

### Regularización
- `num_leaves`: 63 → **47** (modelo más conservador)
- `min_child_samples`: 20 → **40** (Optuna encontró 75 — intermedio)
- Resto de params idénticos a Fase 2

### Patrón OOF vs LB observado
```
           OOF AUC   LB Score   Gap
Fase 2:    0.82480   0.82174    0.003  ← menor gap, mejor generalización
Fase 2b:   0.83830   0.81971    0.019  ← overfit
Fase 2c:   0.83209   pendiente  ???    ← apuntamos a gap < 0.010
```
