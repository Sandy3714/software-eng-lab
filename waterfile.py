# model_waterfall.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Simulated daily temperature data (Days 1-7)
x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([24, 26, 30, 36, 45, 57, 72])  # temperature in °C

# Quadratic model (y = ax^2 + bx + c)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

# Predict temperature for Day 8
prediction = model.predict(poly.transform([[8]]))
print(f"[Waterfall] Predicted temperature on Day 8: {prediction[0]:.2f}°C")

# Plotting
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, model.predict(x_poly), color='red', label='Quadratic Fit')
plt.title("Waterfall Model - Weather Prediction")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()