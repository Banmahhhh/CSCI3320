import pandas as pd
from math import log
from math import sqrt
import math
pd.options.mode.chained_assignment = None


def normal(x, mean, var):
    return -log(sqrt(var))-((x-mean)**2)/(2*var)

data = pd.read_csv('input_3.csv')
length = data.index.size
train = int(length*0.8) # training data length
test = length-train # test data length 
print(length, train, test) 

data_train = data[0:train]
data_test = data[train:length]

# priors of C1 and C2
print("priors of C1 and C2")
prior_c1 = (data_train[data_train['class'] == 1].feature_value.count()) / train
prior_c2 = (data_train[data_train['class'] == 2].feature_value.count()) / train
prior_c3 = (data_train[data_train['class'] == 3].feature_value.count()) / train
prior_c4 = (data_train[data_train['class'] == 4].feature_value.count()) / train
print("prior estimate")
print(prior_c1)
print(prior_c2)
print(prior_c3)
print(prior_c4)

# estimate normal distribution 
m1 = data_train[(data_train['class'] == 1)].feature_value.mean()
var1 = data_train[(data_train['class'] == 1)].feature_value.var()
m2 = data_train[(data_train['class'] == 2)].feature_value.mean()
var2 = data_train[(data_train['class'] == 2)].feature_value.var()
m3 = data_train[(data_train['class'] == 3)].feature_value.mean()
var3 = data_train[(data_train['class'] == 3)].feature_value.var()
m4 = data_train[(data_train['class'] == 4)].feature_value.mean()
var4 = data_train[(data_train['class'] == 4)].feature_value.var()
print("normal distribution")
print(m1, var1)
print(m2, var2)
print(m3, var3)
print(m4, var4)

# test
feature_test = list(data_test['feature_value'])
estimate_test = list()
# print(feature_test)
for i in range(test):
    temp = [normal(feature_test[i], m1, var1)+log(prior_c1), normal(feature_test[i], m2, var2)+log(prior_c2), 
            normal(feature_test[i], m3, var3)+log(prior_c3), normal(feature_test[i], m4, var4)+log(prior_c4)]
    # print(temp)
    # print(data_test.loc[i+train, 'estimate'])
    estimate_test.append(temp.index(max(temp)) + 1)
es = pd.Series(estimate_test)
data_test['estimate'] = es.values
# print(data_test)

confusion_matrix = [[0 for i in range(4)] for j in range(4)]
for i in range(4):
    for j in range(4):
        confusion_matrix[i][j] = data_test[(data_test['estimate'] == j+1) & (data_test['class'] == i+1)].estimate.count()
print("confusion matrix")
print(confusion_matrix)

tp = [0, 0, 0, 0]
tn = [0, 0, 0, 0]
fp = [0, 0, 0, 0]
fn = [0, 0, 0, 0]

for i in range(4):
    tp[i] = data_test[(data_test['class'] == i+1) & (data_test['estimate'] == i+1)].estimate.count()
for i in range(4):
    tn[i] = data_test[(data_test['class'] != i+1) & (data_test['estimate'] != i+1)].estimate.count()
for i in range(4):
    fp[i] = data_test[(data_test['class'] != i+1) & (data_test['estimate'] == i+1)].estimate.count()
for i in range(4):
    fn[i] = data_test[(data_test['class'] == i+1) & (data_test['estimate'] != i+1)].estimate.count()

print("tp, tn, fp, fn")
print(tp)
print(tn)
print(fp)
print(fn)

accuracy = sum(tp) / test
print("accuracy")
print(accuracy)

precision = [0, 0, 0, 0]
recall = [0, 0, 0, 0]
f1 = [0, 0, 0, 0]
for i in range(4):
    if tp[i]+fp[i] == 0:
        precision[i] = float('nan')
    else:
        precision[i] = tp[i] / (tp[i]+fp[i])
    if tp[i]+fn[i] == 0:
        recall[i] = float('nan')
    else:
        recall[i] = tp[i] / (tp[i]+fn[i])
    if math.isnan(precision[i]+recall[i]):
        f1[i] = float('nan')
    else:
        f1[i] = 2*precision[i]*recall[i] / (precision[i]+recall[i])
if math.isnan(sum(f1)):
    macro_f1 = float('nan')
else:
    macro_f1 = sum(f1)/4
print("precision")
print(precision)
print("recall")
print(recall)
print("f1")
print(f1)
print(macro_f1)
