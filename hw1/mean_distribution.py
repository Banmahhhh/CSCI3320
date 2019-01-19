import numpy as np
import matplotlib.pyplot as plt

def estimate_mean(n, delta):
    # code here
    # read the data and reshape
    data = np.loadtxt('/Users/banma/Documents/lovestudying/CSCI3320/hw1/data.txt')
    num = int(data[0])
    rows = int(num/n)
    min = data[1]
    max = data[2]
#     print(num, min, max)
    data = data[3:data.shape[0]]
    data = np.reshape(data, (rows, n))
#     print(data)

    # compute the mean of each row
    data_mean = np.mean(data, axis = 1)
#     print(data_mean)

    # compute PMF
    bins_num = int((max-min)/delta)
#     print(bins_num)

    # generate the list of the keys of the dictionary and initialize
    dict_keys = []
    for i in range(bins_num):
         #print(min+i*delta)
         dict_keys.append(round(min+i*delta, 8))
#     print(dict_keys)
#     print(len(dict_keys))
    bin_pmf_dict = dict.fromkeys(dict_keys, 0)
    # print(bin_pmf_dict)

    # count the number of means in each interval
    for m in data_mean:
         print(m)
         temp = int((m-min)/delta)
         print(temp)
     #     print(round((m-min)/delta, 8))
         bin_pmf_dict[round(min+temp*delta, 8)] += round(1/data_mean.shape[0], 8)
    print(bin_pmf_dict)

    mean_pmf = round(0, 8)
    for i in range(bins_num):
         xi = round(min+delta*i, 8)
     #     print(xi)
         mean_pmf += xi * bin_pmf_dict[xi]
    print(mean_pmf)

#     plt.bar(range(len(D)), list(D.values()), align='center')
    plt.bar(range(len(bin_pmf_dict)), list(bin_pmf_dict.values()))
    plt.title("banma")
    plt.show()
    return bin_pmf_dict, mean_pmf

def main():
     estimate_mean(100, 0.001)

if __name__ == '__main__':
     main()
