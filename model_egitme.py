# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
# from sklearn.model_selection import StratifiedKFold
# from xgboost import XGBClassifier
# import numpy as np
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline

# # ---------------------------
# # 1) Veri yükleme
# # ---------------------------
# train_df = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_training-set.csv")
# test_df  = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_testing-set.csv")

# # ---------------------------
# # 2) Feature - Label ayırma
# # ---------------------------
# label_col = "label"
# # SİLİNMESİ GEREKEN SÜTUNLAR:
# # 'attack_cat': Cevabın ta kendisidir (Sızıntı kaynağı).
# drop_cols = [label_col, "attack_cat"]

# # Hata almamak için sadece veri setinde var olanları sileriz
# existing_drop_cols = [c for c in drop_cols if c in train_df.columns]

# X_train = train_df.drop(existing_drop_cols, axis=1)
# y_train = train_df[label_col]

# # Test setinden de aynılarını siliyoruz
# X_test = test_df.drop(existing_drop_cols, axis=1)
# y_test = test_df[label_col]
# # ---------------------------
# # 3) Kategorik sütunları OneHotEncode
# # ---------------------------
# cat_cols = X_train.select_dtypes(include='object').columns.tolist()
# X_train = pd.get_dummies(X_train, columns=cat_cols)
# X_test  = pd.get_dummies(X_test, columns=cat_cols)
# X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# # ---------------------------
# # 4) Numeric kolonları ölçeklendir
# # ---------------------------
# scaler = StandardScaler()
# num_cols = X_train.select_dtypes(include='number').columns.tolist()
# joblib.dump(num_cols, "xgboost_num_cols.pkl")
# X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
# X_test[num_cols]  = scaler.transform(X_test[num_cols])

# # ---------------------------
# # 5) Class imbalance için weight
# # ---------------------------
# pos = sum(y_train == 1)
# neg = sum(y_train == 0)
# scale_pos_weight = neg / pos

# # ---------------------------
# # 6) Model oluşturma (daha agresif regularization, early stopping olmadan)
# # ---------------------------
# model = XGBClassifier(
#     n_estimators=300,
#     learning_rate=0.005,
#     max_depth=2,
#     subsample=0.6,
#     colsample_bytree=0.6,
#     reg_alpha=10,
#     reg_lambda=10,
#     random_state=42,
#     eval_metric="logloss",
#     scale_pos_weight=scale_pos_weight
# )

# # ---------------------------
# # 7) SMOTE ile pipeline oluştur
# # ---------------------------
# over = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
# pipeline = Pipeline([('over', over), ('model', model)])

# # ---------------------------
# # 8) Cross-validation (Pipeline ile, early stopping olmadan)
# # ---------------------------
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# roc_scores = []

# for train_idx, val_idx in cv.split(X_train, y_train):
#     X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#     y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
#     pipeline.fit(X_tr, y_tr, model__eval_set=[(X_val, y_val)], model__verbose=False)
#     val_pred_prob = pipeline.predict_proba(X_val)[:, 1]
#     roc = roc_auc_score(y_val, val_pred_prob)
#     roc_scores.append(roc)

# print(f"CV ROC-AUC Scores: {roc_scores}")
# print(f"Mean CV ROC-AUC : {np.mean(roc_scores)}")

# # ---------------------------
# # 9) Son modeli tüm eğitim verisi ile eğit
# # ---------------------------
# pipeline.fit(X_train, y_train)

# # ---------------------------
# # 10) Feature Selection (önemli özellikleri tut, overfitting azaltır)
# # ---------------------------
# importances = pipeline['model'].feature_importances_
# important_cols = X_train.columns[importances > 0.005]  # Eşik dene
# print(f"Seçilen özellik sayısı: {len(important_cols)}")

# X_train_selected = X_train[important_cols]
# X_test_selected = X_test[important_cols]
# pipeline.fit(X_train_selected, y_train)

# # ---------------------------
# # 11) Test seti tahmin ve metrikler
# # ---------------------------
# preds = pipeline.predict(X_test_selected)
# pred_probs = pipeline.predict_proba(X_test_selected)[:, 1]

# acc = accuracy_score(y_test, preds)
# prec = precision_score(y_test, preds, zero_division=0)
# rec  = recall_score(y_test, preds, zero_division=0)
# f1   = f1_score(y_test, preds, zero_division=0)
# roc  = roc_auc_score(y_test, pred_probs)

# cm = confusion_matrix(y_test, preds)
# report = classification_report(y_test, preds)

# print("\n============== MODEL PERFORMANSI ==============")
# print("Accuracy :", acc)
# print("Precision:", prec)
# print("Recall   :", rec)
# print("F1 Score :", f1)
# print("ROC-AUC  :", roc)

# print("\n============== CONFUSION MATRIX ==============")
# print(cm)
# print("\n============== CLASSIFICATION REPORT ==============")
# print(report)

# # ---------------------------
# # 12) Model, kolon ve scaler kaydı
# # ---------------------------
# joblib.dump(pipeline['model'], "xgboost_model_regularized.pkl")
# joblib.dump(list(X_train_selected.columns), "xgboost_columns.pkl")
# joblib.dump(scaler, "xgboost_scaler.pkl")
# print("\nModel, kolon ve scaler kaydedildi.")











import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1) Veri yükleme
train_df = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_training-set.csv")
test_df  = pd.read_csv(r"C:\Users\edaul\Desktop\Bitirme_Projesi\NUSW-NB15\UNSW_NB15_testing-set.csv")

# 2) Feature - Label ayırma (Data Leakage Önlemi)
label_col = "label"
drop_cols = [label_col, "id", "attack_cat"] # ID ve Attack_Cat kesinlikle silinmeli
existing_drop_cols = [c for c in drop_cols if c in train_df.columns]

X_train = train_df.drop(existing_drop_cols, axis=1)
y_train = train_df[label_col]

X_test = test_df.drop(existing_drop_cols, axis=1)
y_test = test_df[label_col]

# 3) OneHotEncode
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
X_train = pd.get_dummies(X_train, columns=cat_cols)
X_test  = pd.get_dummies(X_test, columns=cat_cols)
# Sütun eşitleme
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 4) Scaling (ÖNEMLİ: Hangi kolonların scale edildiğini kaydetmeliyiz)
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include='number').columns.tolist()

# Numerik kolon isimlerini kaydediyoruz ki test ederken hangilerini scale edeceğimizi bilelim
joblib.dump(num_cols, "xgboost_num_cols.pkl") 

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# 5) Model Ayarları (Biraz daha agresif/güçlü)
model = XGBClassifier(
    n_estimators=300,           # Ağaç sayısı
    learning_rate=0.05,         #Her ağacın etkisi
    max_depth=6,                #Ağaç derinliği
    subsample=0.8,              #Satır bazlı rastgelelik Verinin % kaçı kullanılır
    colsample_bytree=0.7,       #Sütun bazlı rastgelelik Feature % kaçı kullanılır
    random_state=42,            #Tekrar üretilebilirlik
    eval_metric="logloss",      #Hangi skorla optimize edilecek  logloss ikili sınıflandırma
)


# 6) Pipeline
over = SMOTE(sampling_strategy= 'auto', random_state=42) # verileri dengelemek için bu da iyileştirme parametresi
pipeline = Pipeline([('over', over), ('model', model)])

# 7) Eğit
print("Model eğitiliyor...")
pipeline.fit(X_train, y_train)

# 8) Özellik Seçimi iyileştirme için
importances = pipeline['model'].feature_importances_
# Eşiği düşürdük (0.005 -> 0.001) ki daha fazla özellik kalsın
important_cols_indices = np.where(importances > 0.002)[0] 
important_cols = X_train.columns[important_cols_indices]

print(f"Seçilen özellik sayısı: {len(important_cols)}")

# Seçilenlerle tekrar eğit
X_train_sel = X_train[important_cols]
X_test_sel = X_test[important_cols]

pipeline.fit(X_train_sel, y_train)

# 9) Test Sonuçları
preds = pipeline.predict(X_test_sel)
probs = pipeline.predict_proba(X_test_sel)[:, 1]

print("\n============== EĞİTİM TEST SONUÇLARI ==============")
print("Accuracy :", accuracy_score(y_test, preds))
print("ROC-AUC  :", roc_auc_score(y_test, probs))
print(classification_report(y_test, preds))

# 10) Kayıt
joblib.dump(pipeline['model'], "xgboost_model_regularized.pkl")
joblib.dump(list(X_train_sel.columns), "xgboost_columns.pkl")
joblib.dump(scaler, "xgboost_scaler.pkl")