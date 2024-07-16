import pandas
import numpy
import sklearn
import sklearn.tree
import sklearn.model_selection

DATA_PATH = r"C:\Users\jettd\Documents\ece470\Project\trimmed_data.csv"

def readData():
    print("Reading data...")
    soil = pandas.read_csv(DATA_PATH)
    print(soil)
    classifier = sklearn.tree.DecisionTreeClassifier(max_depth=3)
    X = soil.drop(['score'],axis=1).values
    y = soil['score'].values
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2)
    soilClassifier = classifier.fit(X_train, y_train)
    print(soilClassifier.score(X_test, y_test))

def main():
    readData()

if __name__ == "__main__":
    main()