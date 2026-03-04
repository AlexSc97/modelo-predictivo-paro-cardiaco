import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distributions(df, columns):
    """
    Gráfica la distribución de las variables continuas.
    """
    fig, axes = plt.subplots(len(columns), 2, figsize=(15, 4*len(columns)))
    
    for idx, col in enumerate(columns):
        # Histograma con curva KDE
        sns.histplot(data=df, x=col, kde=True, ax=axes[idx, 0], color='steelblue')
        axes[idx, 0].set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel(col, fontsize=10)
        axes[idx, 0].set_ylabel('Frecuencia', fontsize=10)
        
        # Lineas de media y mediana
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[idx, 0].axvline(mean_val, color='red', linestyle="--", linewidth=2, label=f"Media: {mean_val:.2f}")
        axes[idx, 0].axvline(median_val, color='green', linestyle="--", linewidth=2, label=f"Mediana: {median_val:.2f}")
        axes[idx, 0].legend()
        
        # Boxplots
        sns.boxplot(data=df, x=col, color="lightcoral", ax=axes[idx, 1])
        axes[idx, 1].set_title(f"Box-Plot de: {col}", fontsize='12', fontweight='bold')
        
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importances, feature_names):
    """
    Gráfica la importancia de características a partir de un modelo Random Forest.
    """
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Importancia de las Características (Random Forest)')
    plt.bar(range(len(feature_names)), importances[indices], align='center', color='skyblue')
    plt.xticks(range(len(feature_names)), feature_names[indices], rotation=45)
    plt.xlim([-1, len(feature_names)])
    plt.tight_layout()
    plt.show()
