# Value-Alignment-MAS

Este repositorio contiene la implementación experimental desarrollada para un trabajo de la asignatura **Sistemas Multiagente**, centrado en el problema de la alineación de valores en sistemas multiagente.

El proyecto estudia cómo:

- aprender sistemas de valores individuales a partir de preferencias observadas
- agrupar agentes según sus prioridades éticas
- aplicar restricciones normativas
- analizar cómo afectan dichas normas a distintos grupos sociales

La implementación está inspirada principalmente en el paper:

**Holgado-Sánchez et al. (2025) — Learning the Value Systems of Societies from Preferences**

Sin embargo, este repositorio utiliza una versión simplificada y totalmente reproducible que no requiere ejecutar el entorno original Apollo.

---

## Objetivos del proyecto

El pipeline experimental sigue cuatro etapas principales:

1. Aprendizaje de pesos de valores individuales  
2. Clustering de agentes según sus sistemas de valores  
3. Aplicación de normas  
4. Análisis del impacto normativo  

---

## Dataset

El repositorio incluye una versión procesada del dataset de Apollo:

```text
data/apollo_preferences_dataset.csv
```

Este dataset contiene:

- preferencias entre alternativas
- valores de cada alternativa
- identificadores de comparación
- preferencias a nivel de valor

No es necesario ejecutar el repositorio original de Holgado para reproducir este proyecto.

---

## Estructura del repositorio

```text
Value-Alignment-MAS/
│
├── data/
│   └── apollo_preferences_dataset.csv
│
├── src/
│   ├── learn_value_weights.py
│   ├── evaluate_learning.py
│   ├── cluster_agents.py
│   ├── apply_norms.py
│   ├── analyze_norms_by_cluster.py
│   └── create_plots.py
│
├── outputs/
│   ├── learned_agent_weights.csv
│   ├── learning_evaluation_summary.csv
│   ├── learning_evaluation_by_agent.csv
│   ├── agent_clusters_comparison.csv
│   ├── clustering_summary.csv
│   ├── norms_results.csv
│   └── norms_by_cluster_summary.csv
│
├── graphs/
│
├── memoria/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación

Clonar el repositorio:

```bash
git clone <https://github.com/jcapilla780/Value-Alignment-MAS.git>
cd Value-Alignment-MAS
```

Crear entorno virtual:

```bash
python -m venv myvenv
```

Activarlo:

### Windows

```bash
myvenv\Scripts\activate
```

### Linux / Mac

```bash
source myvenv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Orden de ejecución

### 1. Aprender pesos de valores

```bash
python src/learn_value_weights.py
```

Genera:

```text
outputs/learned_agent_weights.csv
```

---

### 2. Evaluar el modelo de aprendizaje

```bash
python src/evaluate_learning.py
```

Genera:

- `outputs/learning_evaluation_summary.csv`
- `outputs/learning_evaluation_by_agent.csv`

Este paso ejecuta una validación **Leave-One-Out Cross Validation** para evaluar la capacidad predictiva del modelo de regresión logística.

---

### 3. Agrupar agentes

```bash
python src/cluster_agents.py
```

Genera:

- `outputs/agent_clusters_comparison.csv`
- `outputs/clustering_summary.csv`

---

### 4. Aplicar normas

```bash
python src/apply_norms.py
```

Genera:

```text
outputs/norms_results.csv
```

---

### 5. Analizar impacto de normas por cluster

```bash
python src/analyze_norms_by_cluster.py
```

Genera:

```text
outputs/norms_by_cluster_summary.csv
```

---

### 6. Generar gráficos

```bash
python src/create_plots.py
```

Genera los gráficos utilizados en la memoria dentro de:

```text
graphs/
```

---

## Resultados principales

Los principales hallazgos experimentales son:

- El modelo de aprendizaje obtiene un **accuracy medio del 66.6%** mediante validación Leave-One-Out
- Se identifican tres grupos sociales con sistemas de valores diferenciados
- K-Means y clustering jerárquico producen resultados muy similares
- Gaussian Mixture Model obtiene resultados significativamente peores
- Las normas modifican aproximadamente un **4.86%** de las decisiones
- El impacto normativo varía según el grupo social
