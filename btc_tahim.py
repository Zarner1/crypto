# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv("btc_verisi.csv")
df.drop(columns=["Date"], inplace=True)


df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
macd = MACD(close=df["Close"])
df["MACD"] = macd.macd()
df["Signal"] = macd.macd_signal()
df["SMA20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
df["EMA20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()


df = df.iloc[34:].copy()


df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df.dropna(inplace=True)


df.drop(columns=["High", "low"], inplace=True) # CSV'deki sıraya göre 'Close', 'Open', 'Volume' kalır


features = df.drop(columns=["Target"]) # features'ın sütun sırası: ['Close', 'Open', 'Volume', 'RSI', 'MACD', 'Signal', 'SMA20', 'EMA20']
target = df["Target"]


features_columns_for_scaler = features.columns.tolist()
joblib.dump(features_columns_for_scaler, "features_columns_for_scaler.pkl")
print(f"Scaler için özellik sırası kaydedildi: {features_columns_for_scaler}")


scaler = MinMaxScaler()

features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
data_scaled = features_scaled.copy()
data_scaled["Target"] = target.values


X = data_scaled.drop(columns=["Target"])
y = data_scaled["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---- Özellik Seçimi1: RandomForest ve SelectFromModel ----
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

sfm = SelectFromModel(rf, threshold="mean", max_features=5)
sfm.fit(X_train, y_train)
selected_features_sfm = X_train.columns[sfm.get_support()].tolist() 

# ---- Özellik Seçimi2: RFE ----
rfe_selector = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
rfe_selector.fit(X_train, y_train)
rfe_features = X_train.columns[rfe_selector.support_].tolist()

# ---- Özellik Seçimi3: RF Importance ----
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
rf_importance_features = importances.sort_values(ascending=False).head(3).index.tolist() # Değişken adı farklı olsun

# Fonksiyon: 4 model eğit ve değerlendir
def evaluate_models_and_return_metrics(X_train_subset, X_test_subset, y_train, y_test, feature_set_name):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42)
    }
    metrics = []
    cm_dict = {}
    print(f"\n---- {feature_set_name} Özellik Seti ----")
    for name, model in models.items():
        model.fit(X_train_subset, y_train)
        preds = model.predict(X_test_subset) 
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0) 
        
        
        precision = report.get("1", {}).get("precision", 0)
        recall = report.get("1", {}).get("recall", 0)
        f1 = report.get("1", {}).get("f1-score", 0)

        metrics.append((name, accuracy, precision, recall, f1))
        cm = confusion_matrix(y_test, preds)
        cm_dict[name] = cm
        print(f"{name} Doğruluk: {accuracy:.4f}, Hassasiyet: {precision:.4f}, Duyarlılık: {recall:.4f}, F1 Skoru: {f1:.4f}")
        print(f"Confusion Matrix for {name}:\n{cm}\n")
    return metrics, cm_dict


rf_sfm_metrics, rf_sfm_cm = evaluate_models_and_return_metrics(X_train[selected_features_sfm], X_test[selected_features_sfm], y_train, y_test, "RandomForest SelectFromModel")
rfe_metrics, rfe_cm = evaluate_models_and_return_metrics(X_train[rfe_features], X_test[rfe_features], y_train, y_test, "RFE")
rf_importance_metrics, rf_importance_cm = evaluate_models_and_return_metrics(X_train[rf_importance_features], X_test[rf_importance_features], y_train, y_test, "RF Importance")



all_metrics = rf_sfm_metrics + rfe_metrics + rf_importance_metrics
metrics_df = pd.DataFrame(all_metrics, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
metrics_df["Feature Set"] = ["RandomForest SelectFromModel"] * len(rf_sfm_metrics) + \
                            ["RFE"] * len(rfe_metrics) + \
                            ["RF Importance"] * len(rf_importance_metrics)

plt.figure(figsize=(12, 8))
sns.barplot(x="Model", y="Accuracy", hue="Feature Set", data=metrics_df)
plt.title("Model Performansları", fontsize=16)
plt.ylabel("Doğruluk Oranı", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def plot_metrics(metrics_df, metric_name):
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Model", y=metric_name, hue="Feature Set", data=metrics_df)
    plt.title(f"{metric_name} Metrikleri ", fontsize=16)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_metrics(metrics_df, "Precision")
plot_metrics(metrics_df, "Recall")
plot_metrics(metrics_df, "F1 Score")

def plot_confusion_matrix_func(cm, model_name, feature_set_name): 
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title(f"Confusion Matrix: {model_name} ({feature_set_name})")
    plt.ylabel("Gerçek Değerler")
    plt.xlabel("Tahmin Edilen Değerler")
    plt.show()

for feature_set, cm_dict in zip(["RandomForest SelectFromModel", "RFE", "RF Importance"],
                                [rf_sfm_cm, rfe_cm, rf_importance_cm]):
    for model_name, cm_val in cm_dict.items(): 
        plot_confusion_matrix_func(cm_val, model_name, feature_set)



# RFE özellik seti ile Random Forest modelini eğit
rf_rfe_model = RandomForestClassifier(random_state=42)
rf_rfe_model.fit(X_train[rfe_features], y_train)

# Modeli, scaler'ı ve RFE özellik listesini kaydet
joblib.dump(rf_rfe_model, "randomforest_rfe_model.pkl")
joblib.dump(scaler, "scaler.pkl") # Bu zaten var, tekrar kaydetmeye gerek olmayabilir
joblib.dump(rfe_features, "rfe_selected_features.pkl")
joblib.dump(features_columns_for_scaler, "features_columns_for_scaler.pkl") # Bu da zaten var

print("Random Forest (RFE ile) modeli, scaler ve özellik listeleri başarıyla kaydedildi.")
print(f"RFE ile seçilen özellikler: {rfe_features}")