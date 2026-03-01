# Reporte Final: Modelo Predictivo de Paro Cardíaco

## 1. Resumen Ejecutivo
Se ha llevado a cabo de principio a fin un análisis exploratorio y entrenamiento de un modelo de Machine Learning capaz de predecir la presencia o ausencia de enfermedad/paro cardíaco en base a los 14 indicadores clínicos del paciente. 
Los resultados obtenidos demuestran que, usando un algoritmo robusto como *Random Forest Classifier*, podemos tener un porcentaje de predicción confiable, respaldado por una curva ROC y métodos de interpretabilidad modernos (SHAP).

## 2. Metodología Aplicada en el Proyecto
El flujo del proyecto incorporó los siguientes componentes en el entorno de desarrollo:

### 2.1. Exploración Inicial (EDA)
- **Carga de Datos:** Se cargó un dataset histórico con 303 pacientes y las respectivas columnas (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal` y el `target`).
- **Limpieza de Datos:** Verificamos e inspeccionamos la presencia de Nulos, encontrando 0% de nulos. Se removieron líneas duplicadas (1 duplicado encontrado y eliminado con éxito).
- **Estadísticas Analíticas:** Se realizaron sumarios estadísticos y de distribución del objetivo (`target`).

### 2.2. Preprocesamiento de Variables (Data Preprocessing)
- Separamos los datos entre variables independientes ($X$) y la variable dependiente de interés ($y_{target}$).
- **Escalado:** Se utilizó un escalador estándar (`StandardScaler`) para llevar las variables a una distribución normal de media $0$ y desviación estándar de $1$ dentro de la estructura general.
- **División:** El conjunto se dividió en datos de Entrenamiento (80%) y Prueba (20%) utilizando el atributo `stratify=y` que ayuda a mantener el balance poblacional objetivo en ambas partes de nuestros datos.

### 2.3. Entrenamiento del Modelo de ML (Modeling)
- **Pipeline Predictivo:** Construimos un `Pipeline` que integra el escalamiento seguro (previniendo *Data Leakage*) con el modelo Predictivo de Ensamblado: **Random Forest Classifier**.
- **Entrenamiento:** Se entrenó el Random Forest con 100 estimadores (`n_estimators`), una profundidad contenida de 5 ramas (`max_depth=5`) para prevenir sobreajuste (overfitting), y un peso de clases (`class_weight='balanced'`) por si hubiera algún leve desbalance en el target natural.

### 2.4. Evaluación del Modelo Predictivo (Evaluation)
- **Métricas Comunes:** Tras el entrenamiento, obtuvimos un Reporte de Clasificación que evalúa *Precision*, *Recall*, y *F1-score*, logrando separar adecuadamente las predicciones de los pacientes que NO tendrían un evento cardíaco de aquellos que SI tendrían uno.
- **Roc Auc y Matriz:** Se generó la Matriz de Confusión para validar visualmente Falsos Positivos vs. Falsos Negativos y se trazó visualmente la correspondiente Curva ROC-AUC.

### 2.5. Análisis Interpretativo con SHAP
Para hacer que el modelo no sea simplemente una "caja negra" a ojos clínicos o médicos:
- Desplegamos la **Importancia de Características (Feature Importances)** mostrando de forma nativa qué atributos afectan más en Random Forest (e ej. `cp` - tipo de dolor en pecho).
- Implementamos **Modelos SHAP (TreeExplainer)** creando gráficas visuales y un *Summary Plot*, lo que indica claramente como incrementos en ciertos marcadores fisiológicos (e.g. `ca` o depresiones de ondas `oldpeak`) desplazan la predicción o su probabilidad porcentual.

### 2.6. Persistencia del Modelo (Deploy & Persistence)
Con la ejecución final de todo en `explore.ipynb`, terminamos guardando el canal analítico junto al algoritmo predictivo (`pipeline`) gracias a la biblioteca `joblib`.
- **Ruta del artefacto resultante:** `/models/modelo_produccion.pkl`.
Este artefacto será el utilizado en una Interfaz Web o API RESTful que evalúe a los pacientes en tiempo real. 

## 3. Conclusión
El modelo es un éxito, la codificación ha sido almacenada correctamente en el control de versiones y es apto para pasar a una infraestructura productiva. Se recomienda continuar evaluando con datos nuevos (en diferido) que la clínica y especialistas recolecten en el futuro, mediante técnicas de continuous-monitoring y retraining.
