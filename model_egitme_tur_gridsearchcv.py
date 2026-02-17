import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from itertools import cycle

# 1) Verileri Yükle 
print("Veriler yükleniyor...")
train_df = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_training-set.csv")
test_df  = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_testing-set.csv")

# 2) Sadece SALDIRI Verilerini Al (Label=1)
# Hem train hem test setinden normal trafiği çıkarıyoruz, çünkü amacımız tür tahmini.
train_attack = train_df[train_df['label'] == 1].copy()
test_attack  = test_df[test_df['label'] == 1].copy()

print(f"Eğitim Seti Saldırı Sayısı: {len(train_attack)}")
print(f"Test Seti Saldırı Sayısı  : {len(test_attack)}")

# 3) X ve y Hazırla
# Gereksiz sütunları at
drop_cols = ['id', 'label', 'attack_cat']

X_train = train_attack.drop(drop_cols, axis=1)
y_train_raw = train_attack['attack_cat']

X_test = test_attack.drop(drop_cols, axis=1)
y_test_raw = test_attack['attack_cat']

# 4) One-Hot Encoding ve Sütun Eşitleme (KRİTİK ADIM)
X_train = pd.get_dummies(X_train)
X_test  = pd.get_dummies(X_test)

# Test setindeki sütunları Train setiyle birebir aynı sıraya ve sayıya getiriyoruz.
# Eğer test setinde olmayan bir sütun varsa 0 ile doldurulur.
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5) Label Encoding (Hedef Değişken)
le = LabelEncoder()
# Train setine göre öğrenip, hem train'i hem test'i dönüştürüyoruz
y_train = le.fit_transform(y_train_raw)

# Test setinde, train setinde hiç görülmemiş bir saldırı türü varsa hata verebilir.
# UNSW-NB15'te genelde türler ortaktır ama kontrol etmekte fayda var.
# Bilinmeyen türleri yönetmek için basit bir filtreleme
known_classes = set(le.classes_)
test_mask = y_test_raw.isin(known_classes)
X_test = X_test[test_mask]
y_test_raw = y_test_raw[test_mask]
y_test = le.transform(y_test_raw)

# 6) Pipeline Oluşturma
pipeline = Pipeline([
    ('scaler', StandardScaler()),                   
    ('over', SMOTE(sampling_strategy='auto', random_state=42)), 
    ('model', XGBClassifier(
        objective='multi:softprob',               
        num_class=len(le.classes_),               
        eval_metric='mlogloss',                   
        random_state=42,
        n_jobs=-1
    ))
])

# 7) Parametre Izgarası (Grid)
param_grid = {
    'model__n_estimators': [200, 300],      
    'model__max_depth': [6, 8],             
    'model__learning_rate': [0.05, 0.1],    
    'model__subsample': [0.8],              
}

# 8) GridSearch Başlat 
print("\nEn iyi SALDIRI TÜRÜ parametreleri aranıyor...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,                
    scoring='f1_macro',  
    n_jobs=-1,           
    verbose=2
)
grid_search.fit(X_train, y_train)

# 9) En İyi Sonuçlar ve Model Seçimi
print(f"\nEn İyi Parametreler: {grid_search.best_params_}")
print(f"En İyi CV F1 Skoru: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# 10) TEST SETİ İLE TAHMİN
preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)

print("\nTÜR TAHMİNİ SONUÇLARI (TEST SETİ)")
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")
print("\nDetaylı Rapor:")
print(classification_report(y_test, preds, target_names=le.classes_))

# 11) KAYDET
joblib.dump(best_model['model'], "xgboost_type_model.pkl")
joblib.dump(best_model['scaler'], "xgboost_type_scaler.pkl")
joblib.dump(le, "xgboost_type_le.pkl")
joblib.dump(list(X_train.columns), "xgboost_type_columns.pkl")
print(" Model ve yardımcı dosyalar kaydedildi!")

# 12) GÖRSELLEŞTİRME - grafikler
plt.style.use('ggplot')
class_names = le.classes_
n_classes = len(class_names)

# A) Confusion Matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen Tür')
plt.ylabel('Gerçek Tür')
plt.title('Saldırı Türleri Confusion Matrix (Test Seti)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# B) Multiclass ROC Eğrisi
y_test_bin = label_binarize(y_test, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='{0} (AUC = {1:0.2f})'.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Her Saldırı Türü İçin ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()

# C) Feature Importance
xgboost_model = best_model.named_steps['model']
importances = xgboost_model.feature_importances_
feature_names = X_train.columns

feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='magma')
plt.title('Saldırı TÜRÜNÜ Belirleyen En Önemli 20 Özellik')
plt.xlabel('Önem Düzeyi')
plt.tight_layout()
plt.show()