import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro

def test_normality(df, columns):
    """
    Realiza el test de normalidad de Shapiro-Wilk para cada columna.
    H0: Los datos siguen una distribucion normal
    Si p-value  < 0.05 se descarta H0. 
    """
    results = [] 
    for col in columns:
        stats_val, p_value = shapiro(df[col].dropna())
        results.append({
            'Variable': col,
            'Statistic': stats_val,
            'P-value': p_value,
            'Normal': 'Si' if p_value > 0.05 else 'No'
        })
    return pd.DataFrame(results)

def detect_outliers_iqr(df, columns):
    """
    Detecta valores atipicos usando el metodo IQR.
    """
    outliers_summary = []
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outliers_summary.append({
            'Variable': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_bound': lower_bound,
            'Upper_bound': upper_bound,
            'Num_outliers': len(outliers),
            'Outliers_pct': len(outliers) / len(df) * 100
        })
    return pd.DataFrame(outliers_summary)

def detect_outliers_zscore(df, columns, threshold=3):
    """
    Detecta outliers aplicando el metodo z-score.
    """
    outlier_summary = []
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = df[col][z_scores > threshold]

        outlier_summary.append({
            'Variable': col,
            'Mean': df[col].mean(),
            'Std': df[col].std(),
            'Outliers': len(outliers),
            'Outlier_pct': (len(outliers) / len(df)) * 100
        })
    return pd.DataFrame(outlier_summary)
