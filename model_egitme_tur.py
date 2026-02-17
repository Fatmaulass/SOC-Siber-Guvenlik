import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1) Verileri Yükle
train_df = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_training-set.csv")
test_df  = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_testing-set.csv")

# 2) Sadece SALDIRI Verilerini Al
train_attack = train_df[train_df['label'] == 1].copy()
test_attack = test_df[test_df['label'] == 1].copy()

# 3) Gereksizleri At
X_train = train_attack.drop(['id', 'label', 'attack_cat'], axis=1)
y_train_raw = train_attack['attack_cat']

X_test = test_attack.drop(['id', 'label', 'attack_cat'], axis=1)
y_test_raw = test_attack['attack_cat']

# 4) One-Hot Encoding
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Sütunları Eşitle
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5) Attack_Cat → Sayısal
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)

# 6) SMOTE Uygula
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Veri NumPy array'e dönüşmeden önce sütun isimlerini kaydediyoruz!
columns_to_save = list(X_train.columns)

# 7) Ölçekleme
scaler_type = StandardScaler()
# Bu işlemden sonra X_train artık bir Pandas DataFrame değil, NumPy Array olur.
X_train = scaler_type.fit_transform(X_train)
X_test = scaler_type.transform(X_test)

# 8) Multiclass XGBoost Modeli
print("Saldırı Türü Modeli Eğitiliyor...")
model_type = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    objective='multi:softprob',
    num_class=len(le.classes_),
    random_state=42
)

model_type.fit(X_train, y_train)

# 9) Test ve Rapor
print("Test seti üzerinde tahmin yapılıyor")
preds = model_type.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\n (Test Dosyası) Tür Tahmini Başarısı: %{acc*100:.2f}")
print("\nDetaylı Rapor:")
print(classification_report(y_test, preds, target_names=le.classes_))

# 10) Kaydet
joblib.dump(model_type, "xgboost_type_model.pkl")
joblib.dump(scaler_type, "xgboost_type_scaler.pkl")
joblib.dump(le, "xgboost_type_le.pkl")

# Kaydettiğimiz listeyi kullanıyoruz
joblib.dump(columns_to_save, "xgboost_type_columns.pkl") 

print("Tür Tahmin Modeli Kaydedildi!")