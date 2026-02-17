import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

# 1) Veri Yükleme
train_df = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_training-set.csv")
test_df  = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_testing-set.csv")

# 2) Feature - Label Ayrımı
label_col = "label"
drop_cols = [label_col, "id", "attack_cat"]
existing_drop_cols = [c for c in drop_cols if c in train_df.columns]

X_train = train_df.drop(existing_drop_cols, axis=1)
y_train = train_df[label_col]

X_test = test_df.drop(existing_drop_cols, axis=1)
y_test = test_df[label_col]

# 3) One-Hot Encoding
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
X_train = pd.get_dummies(X_train, columns=cat_cols)
X_test  = pd.get_dummies(X_test, columns=cat_cols)

# Sütun eşitleme
X_test = X_test.reindex(columns=X_train.columns, fill_value=0) #Modelin gerçek hayatta hata vermemesi için

# 4) Scaling
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include='number').columns.tolist()

# Numerik kolonları kaydet
joblib.dump(num_cols, "xgboost_num_cols.pkl") 

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# 5) GridSearchCV ile En İyi Parametreleri Bulma

# Temel Model
xgb = XGBClassifier(
    random_state=42, 
    eval_metric="logloss"
)

# Pipeline (SMOTE + Model)
pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy='auto', random_state=42)),
    ('model', xgb)
])

# Denenecek Parametre Izgarası
# Not: Pipeline içindeki model parametrelerine erişmek için 'model__' öneki kullanılır.
param_grid = {
    'model__n_estimators': [100, 200, 300],      # Kaç ağaç olsun?
    'model__max_depth': [4, 6, 8],               # Ağaç derinliği ne olsun?
    'model__learning_rate': [0.01, 0.05, 0.1],   # Öğrenme hızı
    'model__subsample': [0.7, 0.8],              # Verinin ne kadarını kullansın
    'model__colsample_bytree': [0.7, 0.8]        # Sütunların ne kadarını kullansın
}

# GridSearch Ayarları
# cv=3: 3 katlı çapraz doğrulama
# n_jobs=-1: Bilgisayarındaki tüm işlemci çekirdeklerini kullanır
# scoring='roc_auc': En iyi modeli ROC-AUC skoruna göre seçer
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',  #f1 e göre de bakabiliriz
    n_jobs=-1,
    verbose=2
)

# Aramayı Başlat
grid_search.fit(X_train, y_train)

# En iyi parametreleri yazdır
print(f"En İyi Parametreler Bulundu: {grid_search.best_params_}")
print(f" En İyi CV Skoru: {grid_search.best_score_}")

# En iyi modeli al
best_pipeline = grid_search.best_estimator_

# 6) Feature Selection 
importances = best_pipeline.named_steps['model'].feature_importances_

# Eşik değeri (0.002)
important_cols_indices = np.where(importances > 0.002)[0] 
important_cols = X_train.columns[important_cols_indices]

print(f"\nÖzellik Eleme Yapılıyor...")
print(f"Başlangıç Özellik Sayısı: {len(X_train.columns)}")
print(f"Seçilen Özellik Sayısı: {len(important_cols)}")

# Veriyi sadece seçilen özelliklere indirge
X_train_sel = X_train[important_cols]
X_test_sel = X_test[important_cols]

# 7) Final Eğitimi (Seçilen Özellikler ve En İyi Parametrelerle)
# En iyi parametreleri alıp, sadece seçilen özelliklerle yeni bir pipeline kuruyoruz
best_params = grid_search.best_params_

# Parametre isimlerinden 'model__' önekini temizle
clean_params = {k.replace("model__", ""): v for k, v in best_params.items()}

final_model = XGBClassifier(
    **clean_params,
    random_state=42,
    eval_metric="logloss"
)

final_pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy='auto', random_state=42)),
    ('model', final_model)
])

print("Final model (seçilen özelliklerle) eğitiliyor...")
final_pipeline.fit(X_train_sel, y_train)

# 8) Test Sonuçları
preds = final_pipeline.predict(X_test_sel)
probs = final_pipeline.predict_proba(X_test_sel)[:, 1]

print("EĞİTİM TEST SONUÇLARI (OPTİMİZE EDİLMİŞ)")
print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
print(classification_report(y_test, preds))

# 9) Kayıt
joblib.dump(final_pipeline['model'], "xgboost_model_regularized.pkl")
joblib.dump(list(X_train_sel.columns), "xgboost_columns.pkl")
joblib.dump(scaler, "xgboost_scaler.pkl")

# 10) Görselleştirme (Rapor ve Sunum İçin)
plt.style.use('ggplot') # Grafikleri güzelleştirir

# A) Confusion Matrix (Karmaşıklık Matrisi) çizimi
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal', 'Saldırı'],
            yticklabels=['Normal', 'Saldırı'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Durum')
plt.title('Confusion Matrix (Karmaşıklık Matrisi)')
plt.show()

# B) ROC Eğrisi çizimi
fpr, tpr, thresholds = roc_curve(y_test, probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_auc_score(y_test, probs):.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Rastgele tahmin çizgisi
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Alarm Oranı)')
plt.ylabel('True Positive Rate (Saldırı Yakalama Oranı)')
plt.title('ROC (Receiver Operating Characteristic) Eğrisi')
plt.legend(loc="lower right")
plt.show()

# C) Feature Importance (Öznitelik Önem Düzeyleri) çizimi
# Final modelden önem değerlerini al
importances = final_pipeline['model'].feature_importances_
feature_names = X_train_sel.columns

# DataFrame'e çevirip sıralayalım
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(20) # En önemli 20 özellik

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', hue='Feature', legend=False, data=feature_imp_df, palette='viridis')
plt.title('Model İçin En Önemli 20 Özellik (Feature Importance)')
plt.xlabel('Önem Düzeyi')
plt.ylabel('Özellik Adı')
plt.tight_layout()
plt.show()