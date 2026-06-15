"""
Arabic Sentiment Analysis — NLP Pipeline
=========================================
Dataset: train.tsv (all negative) + test.tsv (all positive)
Labels : 'pos' (positive)  |  'neg' (negative)
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import re
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  ARABIC SENTIMENT ANALYSIS")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
neg_df = pd.read_csv(os.path.join(BASE_DIR, "train.tsv"),
                     sep="\t", header=None, names=["label", "text"])
pos_df = pd.read_csv(os.path.join(BASE_DIR, "test.tsv"),
                     sep="\t", header=None, names=["label", "text"])

# Combine and shuffle  (train.tsv = all neg, test.tsv = all pos)
df = pd.concat([neg_df, pos_df], ignore_index=True).sample(frac=1, random_state=42)
df = df.dropna(subset=["text"])

print(f"\n[1] Dataset loaded")
print(f"    Total samples : {len(df):,}")
print(f"    Positive (pos): {(df.label=='pos').sum():,}")
print(f"    Negative (neg): {(df.label=='neg').sum():,}")

# ── 2. Text Preprocessing ─────────────────────────────────────────────────────
ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u065F]")
PUNCTUATION       = re.compile(r"[^\w\s\u0600-\u06FF]")
WHITESPACE        = re.compile(r"\s+")

# Map emojis to sentiment-bearing tokens before stripping
EMOJI_SENTIMENT = {
    # Positive emojis
    "😍": " emoji_love ", "😊": " emoji_happy ", "😁": " emoji_happy ",
    "😂": " emoji_laugh ", "🥰": " emoji_love ", "❤": " emoji_love ",
    "💙": " emoji_love ", "💚": " emoji_love ", "🌹": " emoji_pos ",
    "👏": " emoji_pos ", "🎉": " emoji_pos ", "✨": " emoji_pos ",
    "😄": " emoji_happy ", "🤣": " emoji_laugh ", "😆": " emoji_laugh ",
    "🙂": " emoji_happy ",
    # Negative emojis
    "😢": " emoji_sad ", "😭": " emoji_cry ", "💔": " emoji_broken_heart ",
    "😞": " emoji_sad ", "😔": " emoji_sad ", "😠": " emoji_angry ",
    "😡": " emoji_angry ", "😤": " emoji_angry ", "🌚": " emoji_dark ",
    "😰": " emoji_worried ", "😨": " emoji_scared ", "😟": " emoji_sad ",
}

def preprocess(text: str) -> str:
    """Clean Arabic tweet text while preserving sentiment signals."""
    if not isinstance(text, str):
        return ""
    # 1. Replace emojis with sentiment tokens
    for emoji, token in EMOJI_SENTIMENT.items():
        text = text.replace(emoji, token)
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # 3. Remove mentions and hashtag symbols (keep word)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    # 4. Remove Arabic diacritics (tashkeel)
    text = ARABIC_DIACRITICS.sub("", text)
    # 5. Normalize Arabic letters
    text = re.sub(r"[إأآا]", "ا", text)   # Alef variants → Alef
    text = re.sub(r"ى",      "ي", text)   # Alef maqsura → Ya
    text = re.sub(r"ة",      "ه", text)   # Ta marbuta → Ha
    # 6. Remove non-Arabic / non-space characters (keeps emoji tokens)
    text = re.sub(r"[^\u0600-\u06FF\s_a-z]", " ", text)
    # 7. Collapse whitespace
    text = WHITESPACE.sub(" ", text).strip()
    return text

print("\n[2] Preprocessing text …")
df["clean_text"] = df["text"].apply(preprocess)
df = df[df["clean_text"].str.len() > 2]          # drop empty after cleaning
print(f"    Samples after cleaning: {len(df):,}")

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"\n[3] Train / test split (80 / 20)")
print(f"    Train: {len(X_train):,}   Test: {len(X_test):,}")

# ── 4. Models ─────────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    analyzer="char_wb",   # character n-grams work well for Arabic morphology
    ngram_range=(2, 5),
    max_features=80_000,
    sublinear_tf=True,
    min_df=2,
)

models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Linear SVM":          LinearSVC(C=1.0, max_iter=2000),
}

results = {}
pipelines = {}

print("\n[4] Training models …\n")
print(f"    {'Model':<22} {'Accuracy':>10} {'AUC-ROC':>10}")
print("    " + "-" * 44)

for name, clf in models.items():
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # AUC (only for models that support predict_proba or decision_function)
    try:
        scores = pipe.decision_function(X_test)
    except AttributeError:
        scores = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score((y_test == "pos").astype(int), scores)

    results[name] = {"accuracy": acc, "auc": auc,
                     "y_pred": y_pred, "scores": scores}
    pipelines[name] = pipe
    print(f"    {name:<22} {acc:>9.4f}  {auc:>9.4f}")

# ── 5. Best model detailed report ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best      = results[best_name]

print(f"\n[5] Best model → {best_name}  (accuracy {best['accuracy']:.4f})")
print("\n    Classification Report:")
print(classification_report(y_test, best["y_pred"],
                             target_names=["neg", "pos"],
                             digits=4))

# ── 6. Visualisations ─────────────────────────────────────────────────────────
print("[6] Generating plots …")
os.makedirs("/home/claude/arabic_sentiment/plots", exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"pos": "#4CAF50", "neg": "#E53935"}

# 6a. Label distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["label"].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[PALETTE[l] for l in counts.index], width=0.5, edgecolor="white")
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{v:,}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Label Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count"); ax.set_xlabel("Sentiment")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Negative (neg)", "Positive (pos)"])
plt.tight_layout()
plt.savefig("/home/claude/arabic_sentiment/plots/label_distribution.png", dpi=150)
plt.close()

# 6b. Model accuracy comparison
fig, ax = plt.subplots(figsize=(7, 4))
names = list(results.keys())
accs  = [results[n]["accuracy"] for n in names]
aucs  = [results[n]["auc"]      for n in names]
x     = np.arange(len(names))
w     = 0.35
ax.bar(x - w/2, accs, w, label="Accuracy", color="#1565C0", alpha=0.85)
ax.bar(x + w/2, aucs, w, label="AUC-ROC",  color="#F57F17", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
ax.set_ylim(0.80, 1.0)
ax.set_ylabel("Score"); ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig("/home/claude/arabic_sentiment/plots/model_comparison.png", dpi=150)
plt.close()

# 6c. Confusion matrix for best model
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, best["y_pred"], labels=["neg", "pos"])
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
            xticklabels=["neg", "pos"], yticklabels=["neg", "pos"],
            linewidths=0.5, ax=ax)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("/home/claude/arabic_sentiment/plots/confusion_matrix.png", dpi=150)
plt.close()

# 6d. ROC curve for best model
fig, ax = plt.subplots(figsize=(5, 5))
fpr, tpr, _ = roc_curve((y_test == "pos").astype(int), best["scores"])
ax.plot(fpr, tpr, color="#1565C0", lw=2,
        label=f"AUC = {best['auc']:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve — {best_name}", fontsize=13, fontweight="bold")
ax.legend(loc="lower right"); plt.tight_layout()
plt.savefig("/home/claude/arabic_sentiment/plots/roc_curve.png", dpi=150)
plt.close()

# 6e. Text length distribution
df["text_length"] = df["clean_text"].apply(lambda x: len(x.split()))
fig, ax = plt.subplots(figsize=(7, 4))
for lbl, color in PALETTE.items():
    subset = df[df["label"] == lbl]["text_length"]
    ax.hist(subset, bins=50, alpha=0.65, color=color,
            label=f"{lbl} (mean={subset.mean():.1f})", density=True)
ax.set_xlim(0, 80); ax.set_xlabel("Word Count")
ax.set_ylabel("Density"); ax.set_title("Text Length Distribution", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig("/home/claude/arabic_sentiment/plots/text_length_dist.png", dpi=150)
plt.close()

print("    Plots saved to arabic_sentiment/plots/")

# ── 7. Top TF-IDF features per class ──────────────────────────────────────────
print("\n[7] Top 10 TF-IDF features per class (best model):")
best_pipe   = pipelines[best_name]
vectorizer  = best_pipe.named_steps["tfidf"]
classifier  = best_pipe.named_steps["clf"]

feature_names = np.array(vectorizer.get_feature_names_out())

if hasattr(classifier, "coef_"):
    coef = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
    top_pos_idx = np.argsort(coef)[-10:][::-1]
    top_neg_idx = np.argsort(coef)[:10]
    print(f"\n    {'POSITIVE features':^30}  {'NEGATIVE features':^30}")
    print("    " + "-" * 62)
    for p, n in zip(top_pos_idx, top_neg_idx):
        print(f"    {feature_names[p]:^30}  {feature_names[n]:^30}")

# ── 8. Quick predict function ─────────────────────────────────────────────────
def predict_sentiment(texts, model_name=None):
    """
    Predict sentiment for a list (or single string) of Arabic texts.

    Parameters
    ----------
    texts      : str | list[str]
    model_name : str | None  — defaults to best model

    Returns
    -------
    list of dicts with 'text', 'label', 'confidence'
    """
    if isinstance(texts, str):
        texts = [texts]
    name = model_name or best_name
    pipe = pipelines[name]
    cleaned = [preprocess(t) for t in texts]

    preds = pipe.predict(cleaned)
    try:
        proba = pipe.decision_function(cleaned)
        # Normalise to 0-1 with sigmoid
        confs = 1 / (1 + np.exp(-np.abs(proba)))
    except AttributeError:
        proba = pipe.predict_proba(cleaned)
        confs = proba.max(axis=1)

    return [
        {"text": t, "label": p, "confidence": round(float(c), 4)}
        for t, p, c in zip(texts, preds, confs)
    ]

# Demo predictions
print("\n[8] Demo predictions:")
demo_texts = [
    "الله يستر والله مو قادرة اتخيل",          # should be neg
    "الحمدلله يوم جميل وقلب سعيد",              # should be pos
    "والله زعلت منه كثير ومو راضي",              # should be neg
    "مبروك عليك هذا النجاح يا صديقي",            # should be pos
]
for result in predict_sentiment(demo_texts):
    icon = "✅" if result["label"] == "pos" else "❌"
    print(f"    {icon} [{result['label'].upper()} {result['confidence']:.0%}] "
          f"{result['text'][:50]}")

print("\n" + "=" * 60)
print("  DONE — all results saved to arabic_sentiment/")
print("=" * 60)
