import pickle
import numpy as np
from numpy import load, interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def make_PRC(x, y):
    feature_mat, clinsig_vect = x, y

    # make logistic regression model with l1 regularization
    cv = StratifiedKFold(n_splits=10)
    classifier = LogisticRegression(penalty='l1', solver='liblinear')
    precisions = []
    aucs = []
    f1s = []
    chances = 0
    mean_recalls = np.linspace(0, 1, 100)[:99]
    plt.figure(figsize=(10, 10))
    i = 0

    # randomize order of lesions
    feature_mat, clinsig_vect = shuffle(feature_mat, clinsig_vect)

    for train, test in cv.split(feature_mat, clinsig_vect):
        print("Cross validation #" + str(i))

        # fit logistic regression and predict probabilities
        probas_ = classifier.fit(feature_mat[train], clinsig_vect[train]).predict_proba(feature_mat[test])

        # keep probabilities for positive outcome only
        lr_probs = probas_[:, 1]

        # predict class values
        yhat = classifier.predict(feature_mat[test])

        # calculate precision and recall for each threshold
        lr_precision, lr_recall, thresholds = precision_recall_curve(clinsig_vect[test], lr_probs)
        precisions.append(interp(mean_recalls, lr_precision, lr_recall)[:99])

        # calculate scores
        lr_f1, lr_auc = f1_score(clinsig_vect[test], yhat), auc(lr_recall, lr_precision)
        f1s.append(lr_f1)
        aucs.append(lr_auc)

        # plot chance line
        chance = len(clinsig_vect[test][clinsig_vect[test] == 1]) / len(clinsig_vect[test])
        chances += chance
        #plt.plot([0, 1], [chance, chance], linestyle='--', label='Chance')

        # plot the precision-recall curves
        plt.plot(lr_recall, lr_precision, lw=1, alpha=0.3,
                 label='_nolegend_')
        i += 1

    # calculate and plot mean PRC line
    mean_precision = np.mean(precisions, axis=0)
    mean_auc = auc(mean_recalls, mean_precision)
    std_auc = np.std(aucs)
    mean_f1 = np.mean(f1s)
    plt.plot(mean_recalls, mean_precision, color='b',
             label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f) (F1 = %0.2f)' % (mean_auc, std_auc, mean_f1),
             lw=2, alpha=.8)

    # display standard deviation area
    std_tpr = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_tpr, 1)
    precision_lower = np.maximum(mean_precision - std_tpr, 0)
    plt.fill_between(mean_recalls, precision_lower, precision_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    # plot chances line
    mean_chances = chances/(i+1)
    plt.plot([0, 1], [mean_chances, mean_chances], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    # label graph and show
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall Rate', fontsize=18)
    plt.ylabel('Precision Rate', fontsize=18)
    plt.title('Cross-Validation PRC of Logistic Regression', fontsize=18)
    plt.legend(loc="upper right", prop={'size': 15})
    plt.show()

def make_ROC(x, y):
    feature_mat, clinsig_vect = x, y

    # make logistic regression model with l1 regularization
    cv = StratifiedKFold(n_splits=10)
    classifier = LogisticRegression(penalty='l1', solver='liblinear')
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 10))
    i = 0

    # randomize order of lesions
    feature_mat, clinsig_vect = shuffle(feature_mat, clinsig_vect)

    # 10-fold cross validation splitting
    for train, test in cv.split(feature_mat, clinsig_vect):
        print("Cross validation #" + str(i))

        # fit logistic regression and predict probabilities
        probas_ = classifier.fit(feature_mat[train], clinsig_vect[train]).predict_proba(feature_mat[test])

        # keep probabilities for positive outcome only
        lr_probs = probas_[:, 1]

        # computer roc curve
        fpr, tpr, thresholds = roc_curve(clinsig_vect[test], lr_probs)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        # calculate auc for each fold
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    # plot 0.5 random chance line
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

if __name__ == "__main__":

    #load feature matrix, clinical significance vector, and feature dictionary from files
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

    # show ROC curve
    make_ROC(feature_mat, clinsig_vect)

    # make PRC curve
    make_PRC(feature_mat, clinsig_vect)

