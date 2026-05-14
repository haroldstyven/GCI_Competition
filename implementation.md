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
| **Fase 2** | `phase2_features.ipynb` | **0.82480** | pendiente LB | + TE school/position + z-scores + overall_athleticism |

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

## Fase 3 — Pendiente
- [ ] Hyperparameter tuning con Optuna
- [ ] Ensemble: LGBM + XGB + RF con meta-learner

## Fase 3 — Pendiente
- [ ] Hyperparameter tuning con Optuna
- [ ] Ensemble: LGBM + XGB + RF con meta-learner
