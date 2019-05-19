import pandas as pd
from math import log
from math import sqrt
pd.options.mode.chained_assignment = None


def prob(x1, x2, p, mean, var):
    return x1*log(p) + (1-x1)*log(1-p) - log(sqrt(var)) - ((x2-mean)**2)/(2*var)

data = pd.read_csv('input_4.csv')
length = data.index.size
train = int(length*0.8) # training data length
test = length-train # test data length  

data_train = data[0:train]
data_test = data[train:length]

# priors of C1 and C2
print("priors of C1 and C2")
prior_c1 = (data_train[data_train['class'] == 1].feature_value_1.count()) / train
prior_c2 = (data_train[data_train['class'] == 2].feature_value_1.count()) / train
print("prior estimate")
print(prior_c1)
print(prior_c2)

#estimate distribution
temp_1 = data_train[(data_train['feature_value_1'] == 1) & (data_train['class'] == 1)]
p_1 = temp_1.feature_value_1.count() / ((data_train[data_train['class'] == 1]).feature_value_1.count())
temp_2 = data_train[(data_train['feature_value_1'] == 1) & (data_train['class'] == 2)]
p_2 = temp_2.feature_value_1.count() / ((data_train[data_train['class'] == 2]).feature_value_1.count())
print("p_1, p_2")
print(p_1, p_2)

mean_1 = data_train[(data_train['class'] == 1)].feature_value_2.mean()
var_1 = data_train[(data_train['class'] == 1)].feature_value_2.var()

mean_2 = data_train[(data_train['class'] == 2)].feature_value_2.mean()
var_2 = data_train[(data_train['class'] == 2)].feature_value_2.var()
print("mean_1, var_1, mean_2, var_2")
print(mean_1, var_1, mean_2, var_2)

# test
feature_1 = list(data_test['feature_value_1'])
feature_2 = list(data_test['feature_value_2'])
estimate = list()

for i in range(test):
    if (prob(feature_1[i], feature_2[i], p_1, mean_1, var_1) + log(prior_c1)) > \
        (prob(feature_1[i], feature_2[i], p_2, mean_2, var_2) + log(prior_c2)):
        estimate.append(1)
    else:
        estimate.append(2)

es = pd.Series(estimate)
data_test['estimate'] = es.values
# print(data_test)

tp = data_test[(data_test['class'] == 1) & (data_test['estimate'] == 1)].estimate.count()
fp = data_test[(data_test['class'] == 2) & (data_test['estimate'] == 1)].estimate.count()
tn = data_test[(data_test['class'] == 2) & (data_test['estimate'] == 2)].estimate.count()
fn = data_test[(data_test['class'] == 1) & (data_test['estimate'] == 2)].estimate.count()
confusion_matrix = [[tn, fp], [fn, tp]]
print("confusion matrix:")
print(confusion_matrix)

accuracy = (tp+tn) / test
print("accuracy")
print(accuracy)

precision1 = tp / (tp+fp)
recall1 = tp / (tp+fn)
f1_1 = 2*precision1*recall1 / (precision1+recall1)

precision2 = tn / (tn+fn)
recall2 = tn / (tn+fp)
f1_2 = 2*precision2*recall2 / (precision2+recall2)

macro_f1 = (f1_1 + f1_2) / 2

print(precision1, precision2, recall1, recall2, f1_1, f1_2, macro_f1)