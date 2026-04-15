# Analisis-de-sentimientos-con-DistilBERT-y-Hugging-Face
Análisis de sentimientos a escala sobre el dataset IMDB utilizando Fine-Tuning de DistilBERT con Hugging Face y PyTorch. Logra un 91.6% de accuracy.

Para el análisis de Sentimientos en Reseñas de Cine: Fine-Tuning de DistilBERT
Comprender la carga emocional detrás de la opinión de un usuario es un desafío clásico del Procesamiento de Lenguaje Natural (NLP).
En este proyecto, mi objetivo fue transformar el análisis subjetivo de reseñas de películas en datos clasificables, automatizando la detección de sentimientos (Positivo/Negativo) sobre el célebre dataset de IMDB.
El Desafío del Lenguaje es que el texto presenta retos como el sarcasmo, las dobles negaciones y el contexto cultural. Para abordar esto, utilicé una selección de 12,000 muestras (10k para entrenamiento y 2k para validación) del dataset original
Buscando un modelo que no solo memorice palabras clave, sino que entienda la estructura semántica de una crítica.

La eficiencia de DistilBERT en lugar de entrenar una red neuronal desde cero, implementé Fine-Tuning sobre DistilBERT (distilbert-base-uncased).
¿Por qué DistilBERT? Es una versión destilada, más ligera y rápida del modelo BERT original. Retiene el 97% del rendimiento pero con un 40% menos de parámetros, lo que lo hace ideal, con un balance perfecto entre potencia y consumo de recursos.

El modelo cuenta con una arquitectura de Transformers de 6 capas. Utilicé el tokenizador fast de Hugging Face para procesar secuencias de hasta 512 tokens, aplicando truncamiento y padding para normalizar la entrada de la red.
El proceso de fine-tuning se realizó durante 2 épocas utilizando el optimizador AdamW con una tasa de aprendizaje de $2e-5$ y un decaimiento de peso (weight decay) de 0.01 para prevenir el overfitting.
Utilicé una estrategia de evaluación por época (eval_strategy="epoch") para monitorear el rendimiento en tiempo real.Optimización: Implementé la limpieza de memoria y el mapeo de tensores directamente a GPU (P100/T4 en Kaggle), logrando procesar el entrenamiento en cuestión de minutos.

Resultados y Análisis de Métricas: El modelo alcanzó un Accuracy del 91.65% y un F1-Score de 0.917, resultados sumamente sólidos para un clasificador binario. Análisis del Recall (93.2%)
El modelo demostró ser excepcionalmente bueno identificando reseñas positivas. Casi no deja pasar ninguna opinión favorable sin detectarla.

Al observar la matriz, se nota un equilibrio notable. Sin embargo, existe un pequeño margen de "Falsos Positivos" (99 casos), donde el modelo interpreta como positiva una reseña negativa. Esto suele ocurrir en críticas que mencionan aspectos técnicos buenos (ej: "La fotografía era excelente") pero concluyen que la película fue mala.

El análisis de probabilidades revela que la mayoría de los aciertos ocurren con una confianza cercana al 1.0, mientras que las equivocaciones suelen suceder en el rango de 0.5 a 0.7, lo que indica que el modelo "duda" cuando el lenguaje es ambiguo en lugar de asignar categorías erróneas con total certeza.

Conclusión, este clasificador demuestra que, mediante el Transfer Learning en modelos de lenguaje, es posible alcanzar una fiabilidad cercana a la humana en la interpretación de sentimientos. Aunque el sarcasmo complejo sigue siendo la frontera final, el modelo actual procesa cientos de reseñas por segundo con una precisión del 91%, siendo una herramienta lista para integrarse en pipelines de análisis de datos a gran escala.
<img width="580" height="632" alt="image" src="https://github.com/user-attachments/assets/f489cfb0-eda3-4b9c-ba3d-1fb97e113b40" />
