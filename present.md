---
marp: true
theme: default
paginate: true
math: mathjax
---

# Trabajo Práctico Final

### Integrantes del Grupo

- Lucas Mendoza - lucasfmendoza.99@gmail.com - a1710
- Manuel Pineyro - manuel.o.pineyro@gmail.com - a1819
- José Roberto - jroberto.castrod@gmail.com - e2208
- Ezequiel Caamaño - ezecaa@gmail.com - a1802

---

# Implementación base

---

# Comparación entre LDA y QDA

- LDA asume que cada clase comparte la misma matriz de covarianza
- QDA permite que cada clase tenga su propia matriz de covarianza
- LDA tiene fronteras de decisión lineales
- QDA tiene fronteras de decisión cuadráticas

---

## Implementación: ¿Dónde se aplican estos cambios?

Los cambios en las operaciones de clasificación entre LDA y QDA ocurren en el método de **predicción**.

Flujo de trabajo general:

1. Calcular la media ($\mu_k$) de cada clase.
2. Calcular la covarianza:
   - LDA: compartida $\Sigma$
   - QDA: individual $\Sigma_k$ para cada clase
3. Durante la predicción, calcular $\delta_k(x)$ para cada clase $k$ y seleccionar la clase con el mayor valor.

---

## Implementación: Diferencias clave

En el método `predict`:

- **LDA**: Utiliza una única matriz de covarianza $\Sigma$, calculada una vez y aplicada a todas las clases.

- **QDA**: Aplica la matriz específica $\Sigma_k$ correspondiente a cada clase al calcular $\delta_k(x)$ para cada muestra.

---

# Diferencias entre los datasets Iris y Penguin

**Dataset Iris**

- 150 muestras, 3 clases, 4 características
- Separación clara entre clases, especialmente setosa
- Varianza baja dentro de las clases

**Dataset Penguin**

- 3 especies de pingüinos
- Mayor variabilidad en las características
- Diferencias significativas entre especies

---

# Resultados de la comparación

| Modelo | Dataset | Seed | Error (train) | Error (test) |
| :----: | :-----: | :--: | :-----------: | :----------: |
|  QDA   |  Iris   | 6543 |    0.0222     |    0.0167    |
|  LDA   |  Iris   | 6543 |    0.0222     |    0.0167    |
|  QDA   | Penguin | 6543 |    0.0098     |    0.0073    |
|  LDA   | Penguin | 6543 |    0.0195     |    0.0219    |

---

# Conclusiones de la comparación

- Ambos modelos muestran error de entrenamiento bajo y similar
- QDA tiende a tener mejor desempeño en el conjunto de prueba
- QDA es más flexible al modelar distribuciones Gaussianas
- LDA es robusto cuando las clases tienen estructuras de covarianza similares
- QDA podría ser más propenso a sobreajustarse en datasets pequeños

---

# Optimización Matemática de QDA y LDA

---

# QDA Original, TensorizedQDA y FasterQDA

### QDA Original

- Procesamiento individual de observaciones
- Ineficiente para grandes conjuntos de datos

### TensorizedQDA

- Innovación clave: tensorización (aplicable a ambos)
- Procesamiento simultáneo de múltiples observaciones

---

### FasterQDA (mejoras adicionales):

- Uso de operaciones tensoriales para cálculos más eficientes
- Extracción eficiente de la diagonal
- Método predict optimizado

---

### FasterQDA

- Tensorización y procesamiento simultáneo de múltiples observaciones

```python
inner_prod = unbiased_x.transpose(0, 2, 1) @ self.tensor_inv_cov @ unbiased_x
```

---

### FasterQDA (continuación)

- Extracción de la diagonal

```python
diag_inner_prod = np.stack([np.diagonal(inner_prod[i]) for i in range(inner_prod.shape[0])])
```

---

### FasterQDA: Método predict

```python
def predict(self, X):
    m_obs = X.shape[1]
    y_hat = np.empty(m_obs, dtype=self.encoder.fmt)
    stacked_X = np.stack(X)

    encoded_y_hat_i = np.argmax(self.log_a_priori.reshape(3, 1) + self._predict_log_conditionals(stacked_X), axis=0)

    y_hat = self.encoder.names[encoded_y_hat_i]

    return y_hat
```

---

### Resultados de Tiempo QDA

| Modelo        | Tiempo (ms) |
| ------------- | ----------- |
| QDA           | 1.400       |
| TensorizedQDA | 0.582       |
| FasterQDA     | 0.032       |

FasterQDA es más de 40 veces más rápido que el QDA original!

---

## Propiedad Matricial Clave y EfficientFasterQDA

- Evitamos calcular la matriz n x n completa
- Obtenemos la diagonal de forma eficiente

---

### EfficientFasterQDA

```python
diag_inner_prod = np.sum(unbiased_x_T * temp_product.transpose(0, 2, 1), axis=2)
```

- Multiplica elementos
- Suma filas
- Obtiene la diagonal en una sola línea

---

# Resultados de optimización

| Modelo             | Tiempo promedio de ejecución (s) |
| :----------------- | :------------------------------- |
| QDA                | 0.001201                         |
| TensorizedQDA      | 0.000539                         |
| FasterQDA          | 0.000026                         |
| EfficientFasterQDA | 0.000016                         |

---

## Optimización de LDA

### TensorizedLDA

```python
self.precomputed_product = self.means.transpose(0, 2, 1) @ self.inv_cov
```

- Precálculo para ahorrar tiempo durante la predicción

---

### FasterLDA

```python
log_conditionals = self.precomputed_product @ (X - 0.5 * self.means)
```

- Cálculo en batch de todas las probabilidades
- Procesamiento en paralelo

---

### Resultados de Tiempo LDA

| Modelo        | Tiempo (ms) |
| ------------- | ----------- |
| LDA           | 0.755       |
| TensorizedLDA | 0.271       |
| FasterLDA     | 0.062       |

FasterLDA es más de 12 veces más rápido que el LDA estándar,

---

## Conclusión: optimización matemática

- Mejoras dramáticas en rendimiento para QDA y LDA
- La tensorización y vectorización mejoraron el rendimiento
- El precálculo de valores redujo cálculos redundantes
- Las optimizaciones matemáticas evitaron crear grandes matrices intermedias
- Versiones optimizadas adecuadas para aplicaciones a gran escala y procesamiento en tiempo real

---

# Preguntas teóricas

- Demostración de la función a maximizar en LDA
- Explicación de por qué QDA y LDA son "quadratic" y "linear"
- Diferencias entre la implementación de QDA y la descripción teórica
