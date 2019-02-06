
# Load libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd #Import pandas library
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

results = []
names = []

# Load dataset
def read_data(): #Read data function
    DF = pd.read_csv('bank-full.csv', delimiter=";", \
    true_values=["success","yes"], false_values=["failure","no"])#create data frame from iris.data.txt with pandas.read_csv
    DF = feature_to_dummy(DF, "education", True)
    DF = feature_to_dummy(DF, "marital", True)
    DF = feature_to_dummy(DF, "job", True)
    DF = feature_to_dummy(DF, "contact", True)
    return DF

def feature_to_dummy(DF, column, drop=False):
    tmp = pd.get_dummies(DF[column], prefix=column, prefix_sep='_')
    DF = pd.concat([DF, tmp], axis=1, join_axes=[DF.index])
    if drop:
        del DF[column]
    return DF

def implement_machine_learning(DF): #defined function implement_machine_learning from data frame
    features = ["age", 'job_student', 'job_unemployed',
            "marital_divorced", "marital_married", "marital_single","education_secondary",
                "education_secondary", "education_tertiary", "default","balance","housing", "loan","campaign"]
    validation_size = 0.20
    seed = 7
    scoring = 'accuracy'
    Y = DF['y'] #Selected the y from the column Result
    X = DF[features] #Selected the x to be the features
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Make predictions using KNeighborsClassifier on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    # Test options and evaluation metric
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    # evaluate each model in turn

    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


def compare_algorithms():
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def main(): #main program function
    DF = read_data() #call read data and save data frame to variable
    implement_machine_learning(DF)
    compare_algorithms()

main() #Call the main function
