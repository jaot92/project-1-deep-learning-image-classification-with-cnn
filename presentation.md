---
marp: true
theme: default
class: 
  - lead
  - invert
style: |
  section {
    background: linear-gradient(to bottom right, #1a237e, #0d47a1);
    color: white;
  }
  h1 {
    color: #64ffda;
  }
  h2 {
    color: #40c4ff;
  }
---

# Clasificación de Imágenes con CNN
## Proyecto de Deep Learning
![bg right:40%](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

---

# Índice

1. Introducción y Objetivos
2. Metodología
3. Arquitecturas Implementadas
4. Resultados
5. Análisis Comparativo
6. Conclusiones
7. Demo

---

# 1. Introducción y Objetivos

- **Dataset**: CIFAR-10
  - 60,000 imágenes de 32x32 píxeles
  - 10 clases diferentes
  - 6,000 imágenes por clase

- **Objetivos**:
  - Implementar CNN personalizada
  - Aplicar transfer learning con VGG16
  - Comparar rendimiento de ambos enfoques

---

# 2. Metodología

## Preprocesamiento
- Normalización de píxeles [0,1]
- Data augmentation:
  - Rotación: ±15°
  - Desplazamientos: 10%
  - Volteo horizontal
  - Zoom: ±10%

## Validación
- Split 80/20 train/test
- Early stopping
- Learning rate scheduling

---

# 3. Arquitecturas Implementadas

## CNN Personalizada
![bg right:50% 90%](https://miro.medium.com/max/1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

- 3 bloques convolucionales
- Batch Normalization
- Dropout progresivo
- Dense layers (512 units)

---

# 3. Arquitecturas Implementadas

## Transfer Learning (VGG16)
![bg right:50% 90%](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

- Pesos pre-entrenados ImageNet
- Fine-tuning últimas 4 capas
- Global Average Pooling
- Dense layers personalizadas

---

# 4. Resultados

## CNN Personalizada
- Accuracy: ~75%
- F1-Score: 0.74
- Precisión: 0.75
- Recall: 0.74

## VGG16 Transfer Learning
- Accuracy: ~82%
- F1-Score: 0.81
- Precisión: 0.82
- Recall: 0.81

---

# 5. Análisis Comparativo

![bg right:60% 90%](https://matplotlib.org/3.1.1/_images/sphx_glr_confusion_matrix_001.png)

- VGG16 superó a CNN personalizada
- Mejor generalización
- Convergencia más rápida
- Menor overfitting

---

# 6. Conclusiones

## Ventajas Transfer Learning
- Mayor precisión (+7%)
- Menor tiempo de entrenamiento
- Mejor generalización

## Lecciones Aprendidas
- Importancia del data augmentation
- Efectividad de batch normalization
- Beneficios del fine-tuning

---

# 7. Demo

## API REST con Flask
- Endpoint: `/predict`
- Input: Imagen 32x32
- Output: 
  - Clase predicha
  - Nivel de confianza
- Despliegue local en puerto 5001

---

# ¡Gracias!

## ¿Preguntas?
