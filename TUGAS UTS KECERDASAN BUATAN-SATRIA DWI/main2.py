import re
import argparse
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import os

tqdm.pandas()


def load_public_dataset(name="civilcomment", split="train"):
    """
    Example loaders:
    - Davidson et al. dataset not always packaged; use another public dataset as default.
    - You can replace this loader with local CSV loader.
    """
    # Try load from HF; fallback to local CSV path if provided
    try:
        ds = load_dataset(name, split=split)
        df = pd.DataFrame(ds)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {name}: {e}")


def load_local_csv(path):
    df = pd.read_csv(path)
    return df


def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_pipeline(text, stopwords=set(), stemmer=None, do_stem=False):
    text = basic_clean(text)
    if stopwords:
        text = " ".join([w for w in text.split() if w not in stopwords])
    if do_stem and stemmer is not None:
        text = stemmer.stem(text)
    return text


def prepare_stopwords():
    sf = StopWordRemoverFactory()
    stop = set(sf.get_stop_words())
    # optionally extend stopwords
    return stop


def do_feature_extraction(corpus, method="tfidf", max_features=5000):
    if method == "tfidf":
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        X = tfidf.fit_transform(corpus)
        return X, tfidf
    else:
        raise ValueError("Unknown method")


def train_and_evaluate(X_train, X_test, y_train, y_test, outdir):
    results = {}
    models = {"nb": MultinomialNB(), "logreg": LogisticRegression(max_iter=1000)}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cr = classification_report(y_test, y_pred, target_names=["Tidak Hate", "Hate"])
        cm = confusion_matrix(y_test, y_pred)
        print(f"{name} accuracy: {acc:.4f}")
        print(cr)
        # save model and metrics
        joblib.dump(model, os.path.join(outdir, f"model_{name}.pkl"))
        results[name] = {
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "cr": cr,
            "cm": cm,
        }
        # plot confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred: Tidak Hate", "Pred: Hate"],
            yticklabels=["True: Tidak Hate", "True: Hate"],
        )
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(os.path.join(outdir, f"confusion_{name}.png"))
        plt.close()
    return results


def generate_wordcloud(texts, outpath):
    if not texts:
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(texts)
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_local", action="store_true", help="Load local CSV")
    parser.add_argument(
        "--local_path", type=str, default="indonesian_hate_speech_clean.csv"
    )
    parser.add_argument(
        "--dataset", type=str, default="manueltonneau/indonesian-hate-speech-superset"
    )
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--stem", action="store_true", help="Apply stemming (slower)")
    parser.add_argument("--fast", action="store_true", help="Faster: no stemming")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("1) Load data")
    if args.use_local:
        df = load_local_csv(args.local_path)
    else:
        # if HF path includes owner/dataset, use that; else try fallback
        # NOTE: earlier conversation used manueltonneau dataset which requires auth; for public use swap name
        try:
            # if dataset is local HF id (owner/repo), datasets.load_dataset handles it
            df = pd.DataFrame(load_dataset(args.dataset, split="train"))
        except Exception as e:
            print(
                "Failed to load HF dataset directly, try a common public dataset fallback 'hate_speech18' or local CSV"
            )
            # fallback: user should provide local csv
            raise

    print("Columns:", df.columns.tolist())

    # normalize label column names: try common names
    label_candidates = ["label", "labels", "hate", "target", "class", "y"]
    found = None
    for c in label_candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        raise RuntimeError(
            "No label column found. Required a binary label column (0/1)."
        )
    df = df.rename(columns={found: "label"})
    # ensure numeric 0/1
    df["label"] = df["label"].astype(int)

    print("Label distribution:\n", df["label"].value_counts())

    # Preprocessing
    print("\n2) Preprocessing text")
    stopwords = prepare_stopwords()
    stemmer = None
    if args.stem:
        stem_factory = StemmerFactory()
        stemmer = stem_factory.create_stemmer()

    # progress apply
    def _clean(x):
        return clean_pipeline(
            x, stopwords=stopwords, stemmer=stemmer, do_stem=args.stem
        )

    df["clean_text"] = df["text"].progress_apply(_clean)
    df.to_csv(os.path.join(args.outdir, "cleaned.csv"), index=False)
    print("Saved cleaned.csv")

    # Feature extraction
    print("\n3) Feature extraction (TF-IDF)")
    X, vectorizer = do_feature_extraction(
        df["clean_text"].fillna(""), method="tfidf", max_features=5000
    )
    joblib.dump(vectorizer, os.path.join(args.outdir, "tfidf_vectorizer.pkl"))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Train & evaluate
    print("\n4) Train & Evaluate")
    results = train_and_evaluate(X_train, X_test, y_train, y_test, args.outdir)
    # Wordcloud for hate texts
    hate_texts = df.loc[df["label"] == 1, "clean_text"].tolist()
    generate_wordcloud(hate_texts, os.path.join(args.outdir, "wordcloud_hate.png"))

    # Save summary metrics
    summary = []
    for k, v in results.items():
        summary.append(
            {
                "model": k,
                "acc": v["acc"],
                "prec": v["prec"],
                "rec": v["rec"],
                "f1": v["f1"],
            }
        )
    pd.DataFrame(summary).to_csv(
        os.path.join(args.outdir, "summary_metrics.csv"), index=False
    )
    print("\nDone. Results saved to:", args.outdir)


if __name__ == "__main__":
    main()
