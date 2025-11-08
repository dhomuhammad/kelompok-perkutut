"""
main2.py
End-to-end pipeline for Hate Speech Detection (uses local re_dataset.csv).
Follows assignment steps:
1) Dataset (local)
2) Preprocessing (case-folding, remove punctuation & numbers, tokenization, stopword removal, optional stemming)
3) Feature representation (TF-IDF)
4) Models (Naive Bayes, Logistic Regression, SVM)
5) Evaluation (confusion matrix, accuracy, precision, recall, F1)
6) Visualizations + mini report (report.md)
"""

import os
import re
import string
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# -------------------------
# Config (ubah sesuai keinginan)
# -------------------------
DATA_PATH = "re_dataset.csv"  # file csv di folder yang sama
RESULT_DIR = "results"
RANDOM_STATE = 42
TEST_SIZE = 0.2
USE_STEM = False  # True => pakai Sastrawi stemmer (lebih lambat)
FEATURE_METHOD = "tfidf"  # "tfidf" or "bow"
MAX_FEATURES = 5000
# -------------------------

os.makedirs(RESULT_DIR, exist_ok=True)


# ---------- helper: detect and load ----------
def safe_read_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="latin1")
    return df


if not os.path.exists(DATA_PATH):
    print(f"❌ File tidak ditemukan: {os.path.abspath(DATA_PATH)}")
    raise SystemExit(1)

print("✅ File ditemukan:", os.path.abspath(DATA_PATH))
df_raw = safe_read_csv(DATA_PATH)
print("Loaded. Columns:", list(df_raw.columns))
print("Preview:")
print(df_raw.head(5))

# ---------- map columns (fleksibel) ----------
# Kita cari kolom teks dan kolom label otomatis dari nama-namanya
cols = [c.lower() for c in df_raw.columns]
text_col = None
label_col = None

# kandidat nama teks
text_candidates = [
    "text",
    "tweet",
    "tweet_text",
    "tweettext",
    "isi",
    "teks",
    "utterance",
]
label_candidates = ["label", "labels", "hs", "hate", "target", "y"]

for orig, low in zip(df_raw.columns, cols):
    if low in text_candidates and text_col is None:
        text_col = orig
    if low in label_candidates and label_col is None:
        label_col = orig

# fallback: heuristik if not found
if text_col is None:
    # pilih kolom dengan dtype object dan longest average string length
    obj_cols = [c for c in df_raw.columns if df_raw[c].dtype == object]
    if len(obj_cols) > 0:
        avg_len = {c: df_raw[c].astype(str).map(len).mean() for c in obj_cols}
        text_col = max(avg_len, key=avg_len.get)
        print(f"⚠️ text column autodetected as '{text_col}' (heuristic)")

if label_col is None:
    # coba kolom numeric dengan hanya 0/1 nilai
    num_cols = [c for c in df_raw.columns if np.issubdtype(df_raw[c].dtype, np.number)]
    for c in num_cols:
        uniq = set(df_raw[c].dropna().unique())
        if uniq.issubset({0, 1}) or uniq.issubset({0.0, 1.0}):
            label_col = c
            print(f"⚠️ label column autodetected as '{label_col}' (0/1 numeric)")
            break

if label_col is None:
    raise SystemExit(
        "❌ Gagal menemukan kolom label (0/1). Pastikan dataset memiliki kolom label bernilai 0/1."
    )

print(f"Using text column: '{text_col}'  and label column: '{label_col}'")

# Extract relevant df
df = df_raw[[text_col, label_col]].copy()
df = df.rename(columns={text_col: "text", label_col: "label"})
# Ensure label is int 0/1
df["label"] = df["label"].astype(int)

print("Dataset size:", len(df))
print(df["label"].value_counts())

# ---------- Preprocessing ----------
print("\n=== Preprocessing ===")
nltk.download("stopwords", quiet=True)
# NLTK Indonesian stopwords might not be complete; use NLTK english fallback if needed.
try:
    stop_words = set(stopwords.words("indonesian"))
    if len(stop_words) < 10:
        raise LookupError
except:
    try:
        stop_words = set(stopwords.words("english"))
        print(
            "⚠️ NLTK Indonesian stopwords not available. Fallback to English stopwords."
        )
    except:
        stop_words = set()

# Optionally include simple extra stopwords
extra_stops = {"rt", "user", "url"}
stop_words = stop_words.union(extra_stops)

stemmer = None
if USE_STEM:
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()


def preprocess_one(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()  # case folding
    s = re.sub(r"http\S+|www\S+", " ", s)  # remove urls
    s = re.sub(r"@\w+", " ", s)  # remove mentions
    s = re.sub(r"[^a-z\s]", " ", s)  # remove numbers & punctuation (keep a-z)
    toks = s.split()
    toks = [t for t in toks if t not in stop_words]
    text = " ".join(toks)
    if USE_STEM and stemmer is not None:
        text = stemmer.stem(text)
    return text.strip()


print("Cleaning texts (progress bar)...")
df["clean_text"] = df["text"].progress_apply(preprocess_one)
df.to_csv(os.path.join(RESULT_DIR, "cleaned.csv"), index=False)
print("Saved cleaned.csv")

# ---------- Visualize label distribution ----------
plt.figure(figsize=(5, 4))
sns.countplot(x="label", data=df)
plt.title("Distribusi Label (0 = Tidak Hate, 1 = Hate)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "label_distribution.png"))
plt.close()

# ---------- Word cloud for hate class ----------
hate_corpus = " ".join(df.loc[df["label"] == 1, "clean_text"].astype(str).tolist())
if len(hate_corpus.strip()) > 0:
    wc = WordCloud(width=800, height=400, background_color="white").generate(
        hate_corpus
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud - Hate Speech (label=1)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "wordcloud_hate.png"))
    plt.close()

# ---------- Feature extraction ----------
print("\n=== Feature extraction ===")
if FEATURE_METHOD == "tfidf":
    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
else:
    vec = CountVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))

X = vec.fit_transform(df["clean_text"].fillna(""))
y = df["label"].values

# Save vectorizer
joblib.dump(vec, os.path.join(RESULT_DIR, "vectorizer.pkl"))

# ---------- Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------- Models to train ----------
models = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(),
}

results = []
for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cr = classification_report(y_test, y_pred, target_names=["Tidak Hate", "Hate"])

    print(f"{name} - Acc: {acc:.4f} Prec: {prec:.4f} Rec: {rec:.4f} F1: {f1:.4f}")
    print(cr)

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred:0", "Pred:1"],
        yticklabels=["True:0", "True:1"],
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    img_path = os.path.join(RESULT_DIR, f"confusion_{name}.png")
    plt.savefig(img_path)
    plt.close()

    # save model
    joblib.dump(model, os.path.join(RESULT_DIR, f"model_{name}.pkl"))

    results.append(
        {
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "report": cr,
            "confusion": img_path,
        }
    )

# save metrics table
pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1"]].to_csv(
    os.path.join(RESULT_DIR, "summary_metrics.csv"), index=False
)

# ---------- generate simple markdown report ----------
print("\n=== Generating mini report (report.md) ===")
report_path = os.path.join(RESULT_DIR, "report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Laporan Mini — Deteksi Ujaran Kebencian\n\n")
    f.write("**Dataset**: lokal (`re_dataset.csv`).\n\n")
    f.write(
        "## 1. Latar Belakang\n\nSingkat: tujuan membangun model untuk mendeteksi ujaran kebencian (hate speech) pada teks.\n\n"
    )
    f.write(
        "## 2. Metodologi\n\n- Preprocessing: case folding, hapus URL/mention, hapus tanda baca & angka, tokenisasi dengan .split(), stopword removal (NLTK), stemming (opsional)\n"
    )
    f.write(
        f"- Representasi fitur: {FEATURE_METHOD.upper()}, max_features={MAX_FEATURES}\n"
    )
    f.write("- Model: Naive Bayes, Logistic Regression, LinearSVC\n\n")
    f.write("## 3. Hasil & Analisis\n\n")
    f.write("### Distribusi label\n\n")
    f.write(f"![Distribusi label](label_distribution.png)\n\n")
    if os.path.exists(os.path.join(RESULT_DIR, "wordcloud_hate.png")):
        f.write("### Word Cloud (Hate)\n\n")
        f.write(f"![WordCloud Hate](wordcloud_hate.png)\n\n")
    f.write("### Evaluasi model\n\n")
    for r in results:
        f.write(f"#### Model: {r['model']}\n\n")
        f.write(f"- Accuracy: {r['accuracy']:.4f}\n")
        f.write(f"- Precision: {r['precision']:.4f}\n")
        f.write(f"- Recall: {r['recall']:.4f}\n")
        f.write(f"- F1-score: {r['f1']:.4f}\n\n")
        f.write("Confusion matrix:\n\n")
        # use relative links inside results folder
        imgname = os.path.basename(r["confusion"])
        f.write(f"![Confusion]({imgname})\n\n")
    f.write("## 4. Kesimpulan\n\n")
    f.write(
        "Model baseline berhasil dilatih. Untuk peningkatan, pertimbangkan augmentasi data, embedding (IndoBERT), atau fine-tuning transformer.\n"
    )

print("Report saved to:", report_path)
print("\n===== DONE. Semua file hasil disimpan di folder:", RESULT_DIR, "=====")
