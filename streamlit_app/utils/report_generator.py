import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# 🔹 METRICS
# =========================
def compute_metrics(y_true, y_pred, y_prob=None):
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            results["ROC AUC"] = roc_auc_score(y_true, y_prob)
        except:
            results["ROC AUC"] = None

    return results


# =========================
# 🔹 CSV EXPORT
# =========================
def generate_csv(metrics_dict):
    df = pd.DataFrame(metrics_dict).T
    return df.to_csv(index=True).encode("utf-8")


# =========================
# 🔹 PDF EXPORT
# =========================
def generate_pdf(metrics_dict, filename="report.pdf"):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Software Defect Prediction Report", styles["Title"]))
    content.append(Spacer(1, 12))

    for model_name, metrics in metrics_dict.items():
        content.append(Paragraph(f"<b>{model_name}</b>", styles["Heading2"]))

        for k, v in metrics.items():
            content.append(Paragraph(f"{k}: {round(v, 4)}", styles["Normal"]))

        content.append(Spacer(1, 12))

    doc.build(content)

    return filename