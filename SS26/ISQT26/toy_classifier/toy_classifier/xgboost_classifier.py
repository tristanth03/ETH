import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':

    df = pd.read_excel('C:\Programming\ETH\SS26\ISQT26\\toy_classifier\\toy_classifier\\toy_data.xlsx')
    X = df[['x1', 'x2']].values
    y = df['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    # Train XGBoost model
    xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=10, learning_rate=0.1, max_depth=3)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    # Evaluate both models
    lr_mse = mean_squared_error(y_test, lr_predictions)
    xgb_mse = mean_squared_error(y_test, xgb_predictions)

    print(f"Linear Regression MSE: {lr_mse}")
    print(f"XGBoost MSE: {xgb_mse}")

    # Visualize predictions
    x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
    y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    z = xgb_model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx,yy, z, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('XGBoost')
    plt.show()

    from supertree import SuperTree
    stree = SuperTree(xgb_model, X_train, y_train,)
    stree.show_tree(which_tree=2)
    stree.save_html()