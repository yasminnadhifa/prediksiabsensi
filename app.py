from flask import Flask,redirect,url_for,render_template,request,jsonify
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app=Flask(__name__)
@app.route('/prediksi',methods=['POST'])
def prediksi():
    data = request.get_json()
    tests = data.get('test', [])
    # print(tests)
    # return jsonify({'test': tests})
    # Membaca file iris.csv
    iris = pd.read_csv('static/Book5.csv')
    iris.info()
    # melihat informasi dataset pada 5 baris pertama
    iris.head()
    iris=iris.applymap(str)
    list_of_column_names = list(iris.columns)
    print(iris.columns)
    # memisahkan atribut dan label
    X = iris.loc[:, iris.columns != '31']
    y = iris.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
    # membuat model Decision Tree
    tree_model = DecisionTreeClassifier() 
    # Melatih model dengan menggunakan data latih
    tree_model = tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    acc_secore = round(accuracy_score(y_pred, y_test), 3)
    print('Accuracy: ', acc_secore)
    # prediksi model dengan tree_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
    print(tree_model.predict([tests])[0])
    filename = 'static/model.sav'
    pickle.dump(tree_model, open(filename, 'wb'))
    loaded_model = pickle.load(open('static/model.sav', 'rb'))
    result = loaded_model.predict([tests])[0]
    print(result)
    return jsonify({'data':result})

if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)