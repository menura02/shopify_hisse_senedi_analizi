import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

print("Veri yükleniyor...")
df = pd.read_csv('SHOP_2000-05-20_2025-11-17.csv')

df['close'] = pd.to_numeric(df['close'], errors='coerce')
df = df.dropna(subset=['close'])

df['date'] = pd.to_datetime(df['date'])
df['DateOrdinal'] = df['date'].map(datetime.datetime.toordinal)

X = df[['DateOrdinal']]
y = df['close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)
print(f"Linear Regression R2 Skoru: {r2_lin:.4f}")

degree = 4  
poly_pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Polynomial Regression (Derece {degree}) R2 Skoru: {r2_poly:.4f}")

plt.figure(figsize=(12, 6))

plt.scatter(X, y, color='lightgray', s=10, label='Gerçek Fiyatlar')

X_range = np.linspace(X['DateOrdinal'].min(), X['DateOrdinal'].max(), 500).reshape(-1, 1)

plt.plot(X_range, lin_reg.predict(X_range), color='blue', linewidth=2, label='Linear Model')

plt.plot(X_range, poly_pipeline.predict(X_range), color='red', linewidth=2, label=f'Poly Model (d={degree})')

plt.title(f'Shopify Hisse Trend Analizi: Linear vs Polynomial (d={degree})')
plt.xlabel('Tarih (Ordinal)')
plt.ylabel('Fiyat ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('sonuc_grafigi.png')
print("Grafik 'sonuc_grafigi.png' olarak kaydedildi.")
plt.show()