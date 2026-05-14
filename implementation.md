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
| Baseline | `baseline.ipynb` | ~0.79* | RandomForest, media global, sin flags |
| **Fase 1** | `phase1_lgbm.ipynb` | **0.82056** | LightGBM + imputación grupal + missing flags |

*AUC baseline estimado; no calculado explícitamente en el notebook original.

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

## Fase 2 — Pendiente
- [ ] Target encoding para `School`, `Position`, `Position_Type`
- [ ] Z-scores por posición (rendimiento relativo al grupo)
- [ ] Features compuestas: explosive score, speed-to-size ratio
- [ ] Tendencia temporal (Year)

## Fase 3 — Pendiente
- [ ] Hyperparameter tuning con Optuna
- [ ] Ensemble: LGBM + XGB + RF con meta-learner
