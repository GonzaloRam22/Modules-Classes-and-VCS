import numpy as np
from LogisticRegression import LogisticRegression
from LinearRegression import LinearRegression



if __name__ == "__main__":
    archivo = "Reg_1.txt"
    datos = np.loadtxt(archivo)

    X_linear = datos[:, 0]  # La primera columna es X_linear
    y_linear = datos[:, 1]  # La segunda columna es y_linear

    # Reshape X_linear para que sea una matriz 2D (esto es necesario para la regresión lineal)
    X_linear = X_linear.reshape(-1, 1)

    # Crea una instancia de la clase LinearRegression
    linear_model = LinearRegression(learning_rate=0.01, num_iterations=1000)

    # Entrena el modelo de regresión lineal en los datos de entrada
    linear_model.fit(X_linear, y_linear)

    predictions_linear = linear_model.predict(X_linear)

    # Cálculo de R^2
    mean_y_true = np.mean(y_linear)
    ssr = np.sum((predictions_linear - mean_y_true) ** 2)
    sst = np.sum((y_linear - mean_y_true) ** 2)
    r2 = 1 - (ssr / sst)
    print("R^2:", r2)

    # Calcula el error cuadrático medio (MSE) como métrica de evaluación
    mse = np.mean((y_linear - predictions_linear) ** 2)
    print("Error cuadrático medio (MSE):", mse)

    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Imprime las predicciones
    print("Predicciones de regresión lineal:", predictions_linear)

    print("------------------ modelo 2-------------")

    archivo = "P1_2.txt"
    datos = np.loadtxt(archivo)

    X_logistic = datos[:, :-1]  # Convertir X_logistic en una matriz 2D
    y_logistic = datos[:, -1] - 1

    # Crea una instancia de la clase LogisticRegression
    logistic_model = LogisticRegression(learning_rate=0.0001, num_iterations=1000)

    # Entrena el modelo de regresión logística en los datos de entrada
    logistic_model.fit(X_logistic, y_logistic)

    predictions_logistic = np.array(logistic_model.predict(X_logistic))

    # Calcular Verdaderos Positivos (True Positives)
    tp = np.sum((y_logistic == 1) & (predictions_logistic == 1))

    # Calcular Falsos Positivos (False Positives)
    fp = np.sum((y_logistic == 0) & (predictions_logistic == 1))

    # Calcular Verdaderos Negativos (True Negatives)
    tn = np.sum((y_logistic == 0) & (predictions_logistic == 0))

    # Calcular Falsos Negativos (False Negatives)
    fn = np.sum((y_logistic == 1) & (predictions_logistic == 0))

    print(tp, fp, tn, fn)
    # Calcular Accuracy, Precision y Recall
    accuracy = (tp + tn) / len(y_logistic)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
