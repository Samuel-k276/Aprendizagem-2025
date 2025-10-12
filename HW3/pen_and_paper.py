import numpy as np

X = np.array([[1.0, 4.0], [1.0, 1.0], [1.0, 6.0], [1.0, 18.0], [1.0, 8.0]])
x_train = np.array([4.0, 1.0, 6.0, 18.0, 8.0])
y_train = np.array([3.5, 1.0, 3.8, 10.1, 8.5])
x_test = np.array([0, 12, 5])
y_test = np.array([1, 6.2, 3.6])


def OLS(X, y):
   """
   Computes w = (X^T X)^{-1} X^T y
   """
   w = np.linalg.pinv(X.T @ X) @ X.T @ y
   print(f"\nOLS: y = {w[0]:<.5f} + {w[1]:<.5f} * x")
   return w

def Ridge(X, y):
   """
	Computes w = (X^T X + I)^{-1} X^T y
   """
   w = np.linalg.pinv(X.T @ X + np.eye(X.shape[1])) @ X.T @ y
   print(f"\nRidge: y = {w[0]:<.5f} + {w[1]:<.5f} * x")
   return w

def MAE(y_true, y_pred: callable, x):
   """
   Computes the Mean Absolute Error between true and predicted values.
   """
   y_pred_values = np.asarray([y_pred(xi) for xi in x])
   return np.mean(np.abs(y_true - y_pred_values))

def MAE_analysis(name: str, y_train, y_test, x_train, x_test, model_func):
   """
   Computes and prints the MAE for training and testing datasets.
   """
   MAE_train = MAE(y_train, model_func, x_train)
   MAE_test = MAE(y_test, model_func, x_test)
   print(f"{name:>8} | {MAE_train:<9.5f} | {MAE_test:<.5f}")
   return 

if __name__ == "__main__":
   # Exercicio 1
   w0_OLS, w1_OLS = OLS(X, y_train)
   OLS_function = lambda x: w0_OLS + w1_OLS * x

   # Exercicio 2
   w0_Ridge, w1_Ridge = Ridge(X, y_train)
   Ridge_function = lambda x: w0_Ridge + w1_Ridge * x

   # Exercicio 3
   print("\nApproach | MAE Train | MAE Test")
   MAE_analysis("OLS", y_train, y_test, x_train, x_test, OLS_function)
   MAE_analysis("Ridge", y_train, y_test, x_train, x_test, Ridge_function)
