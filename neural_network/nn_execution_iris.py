import pandas as pd
import neural_network
import matplotlib.pyplot as plt
import numpy as np

iris_df = pd.read_csv("./iris.csv", header=None)
iris_df.columns = ["A","B","C","D","CLASS"]
iris_df = iris_df.sample(frac=1)

iris_df['setosa'] = 0
iris_df['versicolor'] = 0
iris_df['virginica'] = 0

iris_df.loc[iris_df['CLASS'] == 'Iris-setosa', 'setosa'] = 1
iris_df.loc[iris_df['CLASS'] == 'Iris-versicolor', 'versicolor'] = 1
iris_df.loc[iris_df['CLASS'] == 'Iris-virginica', 'virginica'] = 1

iris_df['A'] = iris_df['A'] / iris_df['A'].max()
iris_df['B'] = iris_df['B'] / iris_df['B'].max()
iris_df['C'] = iris_df['C'] / iris_df['C'].max()
iris_df['D'] = iris_df['D'] / iris_df['D'].max()

inputs = iris_df[['A','B','C','D']].values
target = iris_df[['setosa','versicolor','virginica']].values

valid_recs = round(len(inputs)*0.2)
train_recs = len(inputs) - valid_recs

inputs_train = inputs[:train_recs]
target_train = target[:train_recs]

inputs_test = inputs[train_recs:]
target_test = target[train_recs:]

nn = neural_network.neural_network(10)
train_error, validation_error = nn.fit(inputs_train, 
                                       target_train, 
                                       epochs=5000, 
                                       learning_rate=0.03, 
                                       activation=["sigmoid","softmax"], 
                                       validation_data=inputs_test, 
                                       validation_target=target_test, 
                                       correction="batch")

df_erros = pd.DataFrame(train_error)
df_erros.columns = ['train_error']
df_erros['valid_error'] = validation_error
df_erros = df_erros.reset_index()

fig, ax = plt.subplots()
l1, = ax.plot(df_erros['index'].values, df_erros['train_error'].values)
l2, = ax.plot(df_erros['index'].values, df_erros['valid_error'].values)
ax.legend(['Train error', 'Validation Error'])
plt.show()

res = nn.predict(inputs_test)
res = np.array(res)

predictions = np.zeros_like(res)
predictions[np.arange(len(res)), res.argmax(1)] = 1

hit = 0
miss = 0
for i in range(len(predictions)):
    if (target_test[i] == predictions[i]).all():
        hit += 1
    else:
        miss += 1
        
print("Hit: {} Miss: {}".format(hit,miss))
hit_percent = round(hit/len(predictions)*100,2)
miss_percent = round(miss/len(predictions)*100,2)
print("Hit: {}% Miss: {}%".format(hit_percent,miss_percent))