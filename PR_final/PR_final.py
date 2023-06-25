import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import seaborn as sns


# 展示二维高斯分布
def Show(data_path):
    data = np.array(scio.loadmat(data_path)['data']).T
    data, w = PCA(data, 2)
    sns.kdeplot(data[:, 0], data[:, 1], shade=True)
    plt.show()

# PCA降维函数
def PCA(data, d_new):
    n, d = data.shape[0], data.shape[1]
    Data = np.array(data)
    # 所有样本中心化
    for j in range(d):
        Data[:, j] -= np.full(n, np.mean(Data[:, j]))
    # 分解协方差矩阵,取前d_new大特征值对应的特征向量作为投影矩阵W
    c = np.linalg.eig(np.matmul(Data.T, Data) / (n - 1))
    index = np.argsort(-c[0])  # 特征值降序排序
    W = (c[1])[:, index[0:d_new]]
    Data_pca, d = np.matmul(Data, W), d_new
    return Data_pca, W

# PCA分析特征值比例
def PCA_analysis(data_path):
    data = np.array(scio.loadmat(data_path)['data']).T
    n, d = data.shape[0], data.shape[1]
    Data = np.array(data)
    for j in range(d):
        Data[:, j] -= np.full(n, np.mean(Data[:, j]))
    c = np.linalg.eig(np.matmul(Data.T, Data) / (n - 1))
    np.sort(-c[0])  # 特征值降序排序
    lamda_sum_proportion = np.zeros(d)  # 第i个元素表示前i大特征值所占比例和
    lamda_sum_proportion[0] = c[0][0]
    for i in range(1, d):
        lamda_sum_proportion[i] = lamda_sum_proportion[i - 1] + c[0][i]
    lamda_sum_proportion /= np.sum(c[0])

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(d), lamda_sum_proportion)
    plt.xlabel('num of dimensions', fontsize=15)
    plt.ylabel('lamda_sum_proportion', fontsize=15)
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(d), c[0])
    plt.xlabel('dimension', fontsize=15)
    plt.ylabel('lamda', fontsize=15)
    plt.show()


def train(label_data_path, label_path, unlabel_data_path):
    # 数据导入
    label_data = np.array(scio.loadmat(label_data_path)['data']).T       # 有标签样本
    unlabel_data = np.array(scio.loadmat(unlabel_data_path)['data']).T   # 无标签样本
    label = np.array(scio.loadmat(label_path)['label'][0], dtype='int')  # 标签

    data = np.r_[label_data, unlabel_data]  # 有标签在前，无标签在后
    n = data.shape[0]  # 总样本数
    d = data.shape[1]  # 维数
    L_label = len(label_data)      # 有标签样本数
    L_unlabel = len(unlabel_data)  # 无标签样本数
    l = np.zeros(3)   # 确定每一类有标记样本数目
    for i in label:
        l[i] += 1

    # min-max归一化
    for j in range(d):
        Max, Min = max(data[:, j]), min(data[:, j])
        for i in range(n):
            data[i][j] = (data[i][j] - Min) / (Max - Min)

    # 记录训练集每一维的均值
    avr_list = []
    for j in range(d):
        avr_list.append(np.mean(data[:, j]))

    # PCA降维
    data, W = PCA(data, d_new)
    d = d_new

    # 半监督的高斯混合聚类
    # 初始化参数
    μ = [np.zeros(d), np.zeros(d), np.zeros(d)]  # 以i类有标签样本的平均值作为分量i的初始中心
    for i in range(L_label):
        μ[label[i]] += (data[i] / l[label[i]])
    Σ = [np.eye(d), np.eye(d), np.eye(d)]  # 协方差矩阵都初始化为单位矩阵
    a = [1 / 3, 1 / 3, 1 / 3]  # 混合系数初始化为全部相同
    γ = np.zeros((n, 3))  # 后验概率矩阵

    # EM算法迭代，直到满足条件
    time = 0
    while 1:
        time += 1
        print('time:', time)

        # E步：计算后验概率矩阵γ, γ[j][i]表示样本j由第i个高斯分量产生的概率
        for j in range(n):
            p_list = []
            for i in range(3):
                step1 = 1 / np.power(2 * np.pi, d / 2) / np.sqrt(np.abs(np.linalg.det(Σ[i])))
                step2 = np.matmul(data[j] - μ[i], np.linalg.inv(Σ[i]))
                step3 = -0.5 * np.matmul(step2, data[j] - μ[i])
                p_list.append(a[i] * step1 * np.exp(step3))
            γ[j] = p_list / np.sum(p_list)

        # M步：更新均值、协方差矩阵、均值
        μ_new, Σ_new, a_new = [], [], []
        for i in range(3):
            sum1, sum2, sum3, sum6 = 0, 0, 0, 0
            sum4, sum5 = np.zeros((d, d)), np.zeros((d, d))
            # 1.更新均值和混合系数
            for j in range(n):
                if j >= L_label:  # 无标签
                    sum1 += γ[j][i]
                    sum2 += γ[j][i] * data[j]
                if j < L_label and label[j] == i:  # 有标签且类别为i
                    sum3 += data[j]
            μ_new.append((sum2 + sum3) / (sum1 + l[i]))
            a_new.append((sum1 + l[i]) / n)

            # 2.更新协方差矩阵
            for j in range(n):
                if j >= L_label:  # 无标签
                    err = np.mat(data[j] - μ_new[i])
                    sum4 += γ[j][i] * np.matmul(err.T, err)
                if j < L_label and label[j] == i: # 有标签且类别为i
                    err = np.mat(data[j] - μ_new[i])
                    sum5 += np.matmul(err.T, err)
            Σ_new.append((sum4 + sum5) / (sum1 + l[i]))

        # 均值不再变化或超过最大迭代轮数就结束迭代，否则更新参数
        if (np.array(μ) == np.array(μ_new)).all() or time >= MaxTime:
            break
        else:
            μ, Σ, a = μ_new, Σ_new, a_new
    # EM迭代结束

    # 确定样本的簇标记λ,并把样本归入各个分量中
    λ = np.zeros(n)  # 每个样本的簇标记
    Cluster = [[], [], []]  # 簇，每个元素是一个簇（样本编号集）
    for j in range(n):
        λ[j] = np.argmax(γ[j])
        Cluster[int(λ[j])].append(j)

    # 利用有标签样本以多数原则确定高斯分量的类别
    category = []  # category[i]代表高斯分量i的类别
    for i in range(3):
        num = np.zeros(3)
        for j in Cluster[i]:
            if j < L_label:
                num[label[j]] += 1
        category.append(np.argmax(num))

    print('类别顺序：', category)
    return Min, Max, W, μ, Σ, a, category, avr_list


def test(test_data_path, test_label_path, model):
    Min, Max, W, μ, Σ, a, category, avr_list = model

    data = np.array(scio.loadmat(test_data_path)['data']).T
    label = np.array(scio.loadmat(test_label_path)['label'][0], dtype='int')

    n = data.shape[0]  # 测试样本数
    d = data.shape[1]  # 维数

    # 用训练集和测试集的min和max加权得到新的归一化参数
    for j in range(d):
        Max = Max * len_confidence + max(data[:, j]) * (1 - len_confidence)
        Min = Min * len_confidence + min(data[:, j]) * (1 - len_confidence)
        for i in range(n):
            data[i][j] = (data[i][j] - Min) / (Max - Min)

    # 用训练集与测试集的投影参数加权得到新的投影参数
    W_test = PCA(data, d_new)[1]
    for j in range(d):
        avr_mix = avr_list[j] * len_confidence + np.mean(data[:, j]) * (1 - len_confidence)
        data[:, j] -= np.full(n, avr_mix)
    W_mix = W * W_confidence + W_test * (1 - W_confidence)
    data, d = np.matmul(data, W_mix), d_new

    # 给测试样本分类
    label_predict = []
    for j in range(n):
        p_list = []  # 计算每个样本由各个高斯分量生成的概率，取argmax
        for i in range(3):
            step1 = 1 / np.power(2 * np.pi, d / 2) / np.sqrt(np.abs(np.linalg.det(Σ[i])))
            step2 = np.matmul(data[j] - μ[i], np.linalg.inv(Σ[i]))
            step3 = -0.5 * np.matmul(step2, data[j] - μ[i])
            p_list.append(a[i] * step1 * np.exp(step3))
        label_predict.append(category[np.argmax(p_list)])

    ########### 临时 ################
    #sss = [0,0,0]
    #for i in range(n):
       # sss[label_predict[i]] += 1
    #print(sss)
    #acc = 0
    ##################################

    # 对比真实label得出准确率
    num_correct = 0
    for i in range(n):
        num_correct += (label_predict[i] == label[i])
    acc = num_correct / n
    print('准确率：', acc)
    #
    # # 作图展示测试集上分类结果
    # color = ['red', 'blue', 'green']
    # Group = [[], [], []]
    # for i in range(n):
    #     Group[label_predict[i]].append(i)
    # for i in range(3): # 第i类
    #     plt.scatter(data[Group[i]][:, 0], data[Group[i]][:, 1], c=color[i], s=2, label=i) # 展示某两维
    # plt.title('classification result on test set', fontsize=15)
    # plt.legend()
    # plt.show()

    color = ['red', 'blue', 'green']
    Group = [[], [], []]
    for i in range(n):
        Group[label_predict[i]].append(i)
    for i in range(3): # 第i类
        plt.scatter(data[Group[i]][:, 0], data[Group[i]][:, 1], c=color[i], s=2, label=i) # 展示某两维
    plt.title('classification result on test set', fontsize=15)
    plt.legend()
    plt.show()

    return acc



if __name__ == '__main__':
    # 参数
    MaxTime = 20    # 最大迭代轮数(仅不收敛时用）
    d_new = 35      # PCA降到d_new维
    W_confidence = 0.7   # 训练集投影矩阵的权重  # 0.7
    len_confidence = 0    # 训练集长度参数（min，max，avr）的权重  # 0

    model = train('data_train.mat', 'label_train.mat', 'data_unsupervise.mat')  # 训练
    acc = test('data_test.mat', 'label_test.mat', model)  # 测试

