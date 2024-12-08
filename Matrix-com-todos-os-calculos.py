import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

np.random.seed(42)
y_true = np.random.randint(0, 10, size=100)  
y_pred = np.random.randint(0, 10, size=100)  


matriz_confusao = confusion_matrix(y_true, y_pred, labels=np.arange(10))

# Exibir a matriz 
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues, values_format="d")  
plt.title("Matriz de Confusão")
plt.figure(figsize=(12, 8))  
plt.show()

#  métricas para  classe
def calcular_metricas(matriz_confusao):
    metrics = {}
    for i in range(10):
        VP = matriz_confusao[i, i] 
        FN = np.sum(matriz_confusao[i, :]) - VP 
        FP = np.sum(matriz_confusao[:, i]) - VP  
        VN = np.sum(matriz_confusao) - (VP + FN + FP)  
        acuracia = (VP + VN) / (VP + VN + FP + FN)
        sensibilidade = VP / (VP + FN) if (VP + FN) != 0 else 0
        especificidade = VN / (VN + FP) if (VN + FP) != 0 else 0
        precisao = VP / (VP + FP) if (VP + FP) != 0 else 0
        f_score = (2 * precisao * sensibilidade) / (precisao + sensibilidade) if (precisao + sensibilidade) != 0 else 0

        metrics[i] = {
            "Acurácia": acuracia,
            "Sensibilidade (Recall)": sensibilidade,
            "Especificidade": especificidade,
            "Precisão": precisao,
            "F-Score": f_score
        }

    return metrics

# Calcular as métricas
metricas = calcular_metricas(matriz_confusao)
for i in range(10):
    print(f"\nMétricas para a Classe {i}:")
    for metrica, valor in metricas[i].items():
        print(f"{metrica}: {valor:.2f}")


y_true_bin = label_binarize(y_true, classes=np.arange(10))
y_pred_bin = label_binarize(y_pred, classes=np.arange(10))

#  Curva ROC
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para o modelo de classificação')
plt.legend(loc="lower right")
plt.show()
