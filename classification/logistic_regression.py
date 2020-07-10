import pickle
import numpy as np
from numpy import load, interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file1 ='C:\\Users\\haoli\\Documents\\pcavision\\feature_extraction\\feature_mat.npy'
    feature_mat = load(file1)
    print("Reading back feature matrix")
    print(feature_mat)
    file2 = 'C:\\Users\\haoli\\Documents\\pcavision\\feature_extraction\\clinsig_vect.npy'
    clinsig_vect = load(file2)
    print("Reading back clinical significance vector")
    print(clinsig_vect)
    file3 = 'C:\\Users\\haoli\\Documents\\pcavision\\feature_extraction\\feature_dict.txt'
    with open(file3, 'rb') as handle:
        feature_dict = pickle.loads(handle.read())
    print("Reading back feature dict")
    print(feature_dict)

    # make logistic regression model with l1 regularization and train/test 80/20 split
    #x_train, x_test, y_train, y_test = train_test_split(feature_mat, clinsig_vect, test_size=0.2, random_state=0)
    # compare penalty='l1' with no penalty
    # predicted = model_selection.cross_val_predict(LogisticRegression(penalty='l1', solver='liblinear'), feature_mat, clinsig_vect, cv=10)
    # print(metrics.classification_report(clinsig_vect, predicted))


    cv = StratifiedKFold(n_splits=10)
    classifier = LogisticRegression(penalty='l1', solver='liblinear')

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 10))
    i = 0
    for train, test in cv.split(feature_mat, clinsig_vect):
        print("Cross validation #" + str(i))
        probas_ = classifier.fit(feature_mat[train], clinsig_vect[train]).predict_proba(feature_mat[test])
        #compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(clinsig_vect[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Cross-Validation ROC of Logistic Regression', fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()
