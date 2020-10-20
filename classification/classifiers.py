import pickle
import numpy as np
from numpy import load, interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import xgboost as xgb
from scipy.stats import uniform, randint, norm
import statsmodels.api as sm


def make_test_PRC(X_train, y_train, X_test, y_test):
    # fit a model
    # model = LogisticRegression(penalty='l1', solver='liblinear')
    # model = sklearn.svm.SVC(C=0.001, kernel='rbf', probability=True)
    # model = RandomForestClassifier(criterion='entropy', n_estimators=1000)
    model = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.7302334004132279, gamma=0.009110912825774864, learning_rate=0.05833288822677851, max_depth=4, n_estimators=122, subsample=0.6284754593840916)
    model.fit(X_train, y_train)

    # predict probabilities
    lr_probs = model.predict_proba(X_test)

    # keep probabilities for positive outcome only
    lr_probs = lr_probs[:, 1]

    # predict class values
    yhat = model.predict(X_test)
    lr_precision, lr_recall, thresholds = precision_recall_curve(y_test, lr_probs)
    lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

    # calculate and plot mean PRC line
    plt.plot(lr_recall, lr_precision, color='b',
             label=r'Mean PRC (AUC = %0.2f) (F1 = %0.2f)' % (lr_auc, lr_f1),
             lw=2, alpha=.8)

    # plot chances line
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    # label graph and show
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall Rate', fontsize=18)
    plt.ylabel('Precision Rate', fontsize=18)
    plt.title('Test Set PRC of XGBoost', fontsize=18)
    plt.legend(loc="lower left", prop={'size': 15})
    plt.show()
    print(classification_report(y_test, yhat))

    importance = model.feature_importances_ 
    plt.bar([x for x in range(len(importance[26:]))], importance[26:])
    plt.show()

    print(lr_precision[121], lr_recall[121], thresholds[121])
def make_train_PRC(x, y):
    feature_mat, clinsig_vect = x, y

    # make logistic regression model with l1 regularization
    cv = StratifiedKFold(n_splits=10)
    classifier = LogisticRegression(penalty='l1', solver='liblinear')
    #classifier = sklearn.svm.SVC(C=0.00000000001, kernel='rbf', probability=True)
    #classifier = RandomForestClassifier(criterion='entropy')
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
    plt.legend(loc="lower left", prop={'size': 15})
    plt.show()

def make_test_ROC(X_train, y_train, X_test, y_test):

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # fit a model
    # model = LogisticRegression(penalty='l1', solver='liblinear')
    # model = sklearn.svm.SVC(C=0.001, kernel='rbf', probability=True)
    # model = RandomForestClassifier(criterion='entropy', n_estimators=1000)
    model = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.7135656010318567, gamma=0.013183487248626002, learning_rate=0.14293901006341486, max_depth=2, n_estimators=117, subsample=0.9949104517259778)
    model.fit(X_train, y_train)

    # predict probabilities
    lr_probs = model.predict_proba(X_test)
    print(type(lr_probs))
    # keep probabilities for positive outcome only
    lr_probs = lr_probs[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)

    # calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    # plot roc curve for model
    plt.plot(lr_fpr, lr_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f)' % lr_auc,
             lw=2, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Test Set ROC of XGBoost', fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()

    # print(model.coef_, model.intercept_)

def make_train_ROC(x, y):
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

def search_hyperparameter(X, y):
    xgb_model = xgb.XGBRegressor()

    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 6),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1,
                                n_jobs=1, return_train_score=True)

    search.fit(X, y)

    report_best_scores(search.cv_results_, 1)

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == "__main__":

    #load feature matrix, clinical significance vector, and feature dictionary from files
    file1 ='C:\\Users\\haoli\\Documents\\pcavision\\feature_extraction\\PZ\\train_feature_mat_pz.npy'
    feature_mat = load(file1)
    print("Reading back feature matrix")
    print(np.shape(feature_mat))


    file2 = 'C:\\Users\\haoli\\Documents\\pcavision\\feature_extraction\\PZ\\train_clinsig_vect_pz.npy'
    clinsig_vect = load(file2)
    print("Reading back clinical significance vector")
    print(np.shape(clinsig_vect))

    # file3 = 'C:\\Users\\haoli\\Documents\\pcavision\\feature_extraction\\PZ\\allfeature_dict.txt'
    # with open(file3, 'rb') as handle:
    #     feature_dict = pickle.loads(handle.read())
    # print("Reading back feature dict")

    file4 = 'C:\\Users\\haoli\\Documents\\pcavision\\feature_extraction\\PZ\\test_feature_mat_pz.npy'
    # \\test_feature_mat_pz.npy
    test_feature_mat = load(file4)
    print("Reading back testing feature matrix")

    file5 = 'C:\\Users\\haoli\\Documents\\pcavision\\feature_extraction\\PZ\\test_clinsig_vect_pz.npy'
    test_clinsig_vect = load(file5)
    print()

    feature_mat = np.nan_to_num(feature_mat)
    test_feature_mat = np.nan_to_num(test_feature_mat)
    # show ROC curve for training cross validation
    # make_train_ROC(feature_mat, clinsig_vect)

    #make PRC curve for training cross validation
    #make_train_PRC(feature_mat, clinsig_vect)

    # make ROC curve for test
    #make_test_ROC(feature_mat, clinsig_vect, test_feature_mat, test_clinsig_vect)

    # make PRC curve for test and print coefficient vector
    make_test_PRC(feature_mat, clinsig_vect, test_feature_mat, test_clinsig_vect)

    # get optimal hyperparameters for XGBoost
    # search_hyperparameter(feature_mat, clinsig_vect)
