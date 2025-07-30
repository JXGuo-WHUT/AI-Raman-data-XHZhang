from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=1000, n_classes=3, n_features=20, n_informative=10, random_state=42)


y_bin = label_binarize(y, classes=[0, 1, 2])


X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)


model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train, y_train)


y_score = model.predict_proba(X_test)


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for 3-Class Problem (One-vs-Rest)')
plt.legend(loc="lower right")
plt.show()