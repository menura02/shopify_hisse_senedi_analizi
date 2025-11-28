
# Giriş

Bu proje, Basit Doğrusal Regresyon (Linear Regression) ve Polinom Regresyon (Polynomial Regression) tekniklerini kullanarak Shopify hisse senedi fiyat trendini modellemeyi amaçlar.

Çalışma boyunca kullandığım kütüphaneler:

- Pandas: Veriyi temizlemek ve düzenlemek için.
- Matplotlib: Sonuçları görselleştirmek için.
- Scikit-learn: Regresyon modellerini oluşturmak ve eğitmek için

---

# 1. Veri Hazırlığı

Öncelikle veri setini yükleyip modele uygun hale getirmeliyiz. Veri setinde bazı satırlarda sayısal değerler yerine metin hataları bulunuyor, bu yüzden **öncelikle temizlik işlemi yapmalıyız**. Ayrıca regresyon modelleri tarih formatındaki verilerle doğrudan matematiksel işlem yapamadığı için tarihleri sayısal bir formata çevirmeliyiz.

```python
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Veriyi yükle
df = pd.read_csv('SHOP_2000-05-20_2025-11-17.csv')

# Veri Temizliği
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df = df.dropna(subset=['close'])

# Tarih formatını ayarla ve Sayısal Değere çevir
df['date'] = pd.to_datetime(df['date'])
df['DateOrdinal'] = df['date'].map(datetime.datetime.toordinal)

# Hedef (y) ve Öznitelik (X) belirle
X = df[['DateOrdinal']]
y = df['close']
````

---

# 2. Doğrusal Regresyon Denemesi

İlk olarak verilere düz bir çizgi uydurmayı denedim. **(Doğrusal regresyon, veriler arasındaki ilişkiyi y=mx+b denklemiyle ifade etmeye çalışır.)**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Veriyi Eğitim (%80) ve Test (%20) olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Tahmin yap
y_pred_lin = lin_reg.predict(X_test)

# Skoru yazdır
print(f"Linear Regression R2: {r2_score(y_test, y_pred_lin):.4f}")
```

Gözlem: Doğrusal model hissenin genel olarak arttığını gösteren yukarı yönlü bir çizgi çizdi. Ancak hisse senedindeki dalgalanmaları, zirve ve dip noktalarını tamamen göz ardı etti (Underfitting). Kısaca model bize şunu söylüyor: **"Zaman geçtikçe fiyat dalgalanma olmadan, sürekli artar."** .

---

# 3. Polinom Regresyon Denemesi

Grafikte verilerimiz düz bir çizgi gibi hareket etmediği için, verideki eğrileri ve dalgalanmaları yakalayabileceğini düşündüğüm Polinom Regresyon'u denemeye karar verdim.

Scikit-learn kütüphanesindeki PolynomialFeatures aracını kullanarak tarih verisinin 4. dereceden üslerini aldım. **Bu dereceyi seçmemimin nedeni**, Shopify hissesinin "Yükseliş -> Zirve -> Düşüş -> Toparlanma" şeklindeki karmaşık döngüsünü yakalamaktı. **Dereceyi 50 gibi yüksek bir sayı yapsaydım** eğitim başarısı %99'a yakın bir değer olurdu ama bu **overfitting yaşanmasına sebebiyet verirdi**. Ufak bir tarih değişikliğinde yüksek fiyat değişikliklerine sebebiyet verirdi.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 4. Dereceden bir polinom oluştur
degree = 4
poly_pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Modeli eğit
poly_pipeline.fit(X_train, y_train)

# Tahmin yap
y_pred_poly = poly_pipeline.predict(X_test)

# Skoru yazdır
print(f"Polynomial Regression R2: {r2_score(y_test, y_pred_poly):.4f}")
```

---

# 4. Karşılaştırma

İki yöntemi de karşılaştırdım ve **en uygun çözümün Polinom Regresyon olduğuna karar verdim**. Ama borsa verileri hem rastgelelik içeren veriler olduğu için hem de hissenin hareketleri sadece zamanın geçmesiyle açıklanamadığı için tahminlerimiz kusurlu çıkıyor.

| Model                       | R2 Skoru (Başarım) | Gözlem                                                                        |
| --------------------------- | ------------------ | ----------------------------------------------------------------------------- |
| Doğrusal Regresyon          | ~0.49              | Sadece genel yönü biliyor, dalgalanmaları kaçırıyor.               |
| Polinom Regresyon           | ~0.51              | Fiyatın parabolik hareketlerini ve trend değişimlerini daha iyi takip ediyor. |

---

# 5. Veri Görselleştirme

Matplotlib kütüphanesi kullanarak verileri görselleştirdim. 

Görselleştirdiğim veriler:
- Gri Noktalar: Gerçekleşmiş Shopify kapanış fiyatları.
- Mavi Çizgi: Doğrusal Regresyon modelinin tahmin ettiği veriler.
- Kırmızı Eğri: Polinom Regresyon modelinin tahmin ettiği veriler.

<img width="1200" height="600" alt="sonuc_grafigi" src="https://github.com/user-attachments/assets/64a9ddeb-1b8f-42d5-994d-85187caa7d03" />



```
```
