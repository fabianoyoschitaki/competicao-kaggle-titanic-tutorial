
.. code:: ipython3

    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

.. code:: ipython3

    #Lendo conjuntos de teste e treino
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

.. code:: ipython3

    #Cinco primeiras linhas do conjunto de treino
    train.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>PassengerId</th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Name</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Ticket</th>
          <th>Fare</th>
          <th>Cabin</th>
          <th>Embarked</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>Braund, Mr. Owen Harris</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>A/5 21171</td>
          <td>7.2500</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>PC 17599</td>
          <td>71.2833</td>
          <td>C85</td>
          <td>C</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>1</td>
          <td>3</td>
          <td>Heikkinen, Miss. Laina</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>STON/O2. 3101282</td>
          <td>7.9250</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
          <td>female</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>113803</td>
          <td>53.1000</td>
          <td>C123</td>
          <td>S</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>0</td>
          <td>3</td>
          <td>Allen, Mr. William Henry</td>
          <td>male</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>373450</td>
          <td>8.0500</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    #Retirando colunas com nome, ingresso e cabine dos conjuntos
    train.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
    test.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

.. code:: ipython3

    #Sem as 3 colunas
    train.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>PassengerId</th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Fare</th>
          <th>Embarked</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>7.2500</td>
          <td>S</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>71.2833</td>
          <td>C</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>1</td>
          <td>3</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>7.9250</td>
          <td>S</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>female</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>53.1000</td>
          <td>S</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>0</td>
          <td>3</td>
          <td>male</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>8.0500</td>
          <td>S</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    #Criação de novo DataFrame a partir de One-Hot encoding
    new_data_train = pd.get_dummies(train)
    new_data_test = pd.get_dummies(test)

.. code:: ipython3

    #Primeiras linhas do novo dataset de teste
    new_data_test.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>PassengerId</th>
          <th>Pclass</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Fare</th>
          <th>Sex_female</th>
          <th>Sex_male</th>
          <th>Embarked_C</th>
          <th>Embarked_Q</th>
          <th>Embarked_S</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>892</td>
          <td>3</td>
          <td>34.5</td>
          <td>0</td>
          <td>0</td>
          <td>7.8292</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>893</td>
          <td>3</td>
          <td>47.0</td>
          <td>1</td>
          <td>0</td>
          <td>7.0000</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>894</td>
          <td>2</td>
          <td>62.0</td>
          <td>0</td>
          <td>0</td>
          <td>9.6875</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>895</td>
          <td>3</td>
          <td>27.0</td>
          <td>0</td>
          <td>0</td>
          <td>8.6625</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>896</td>
          <td>3</td>
          <td>22.0</td>
          <td>1</td>
          <td>1</td>
          <td>12.2875</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    #Primeiras linhas do novo dataset de treino
    new_data_train.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>PassengerId</th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Fare</th>
          <th>Sex_female</th>
          <th>Sex_male</th>
          <th>Embarked_C</th>
          <th>Embarked_Q</th>
          <th>Embarked_S</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>7.2500</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>71.2833</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>1</td>
          <td>3</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>7.9250</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>53.1000</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>0</td>
          <td>3</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>8.0500</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    #Quantidade de valores nulos no conjunto de treino
    new_data_train.isnull().sum().sort_values(ascending=False).head(10)




.. parsed-literal::

    Age           177
    Embarked_S      0
    Embarked_Q      0
    Embarked_C      0
    Sex_male        0
    Sex_female      0
    Fare            0
    Parch           0
    SibSp           0
    Pclass          0
    dtype: int64



.. code:: ipython3

    #Preenchendo valores nulos
    new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
    new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)

.. code:: ipython3

    #Quantidade de valores nulos no conjunto de treino
    new_data_train.isnull().sum().sort_values(ascending=False).head(10)




.. parsed-literal::

    Embarked_S    0
    Embarked_Q    0
    Embarked_C    0
    Sex_male      0
    Sex_female    0
    Fare          0
    Parch         0
    SibSp         0
    Age           0
    Pclass        0
    dtype: int64



.. code:: ipython3

    #Quantidade de valores nulos no conjunto de Teste
    new_data_test.isnull().sum().sort_values(ascending=False).head(10)




.. parsed-literal::

    Fare          1
    Embarked_S    0
    Embarked_Q    0
    Embarked_C    0
    Sex_male      0
    Sex_female    0
    Parch         0
    SibSp         0
    Age           0
    Pclass        0
    dtype: int64



.. code:: ipython3

    #Preenchendo valores nulos 'Fare'
    new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)

.. code:: ipython3

    #Separando features e target para criação do modelo
    x = new_data_train.drop('Survived', axis=1)
    y = new_data_train['Survived']

.. code:: ipython3

    #Features não contém o campo survived
    x.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>PassengerId</th>
          <th>Pclass</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Fare</th>
          <th>Sex_female</th>
          <th>Sex_male</th>
          <th>Embarked_C</th>
          <th>Embarked_Q</th>
          <th>Embarked_S</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>3</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>7.2500</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>71.2833</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>3</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>7.9250</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>1</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>53.1000</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>3</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>8.0500</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    #Target contém somente a coluna Survived
    y.head()




.. parsed-literal::

    0    0
    1    1
    2    1
    3    1
    4    0
    Name: Survived, dtype: int64



.. code:: ipython3

    #Criação do modelo e treino
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(x, y)




.. parsed-literal::

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=0,
                splitter='best')



.. code:: ipython3

    #Verificando score no conjunto de treino
    tree.score(x, y)




.. parsed-literal::

    0.8271604938271605



.. code:: ipython3

    #Você precisa enviar ao Kaggle um arquivo .csv com as suas previsões. 
    #Esse arquivo deve conter duas colunas com cabeçalho: PassengerId e Survived.
    submission = pd.DataFrame()
    submission['PassengerId'] = new_data_test['PassengerId']
    submission['Survived'] = tree.predict(new_data_test)
    submission.to_csv('submission.csv', index=False)
