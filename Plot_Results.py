import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix
from itertools import cycle

import seaborn as sns


def Plot_Confusion():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)[:, :, :, :, :, 0]
    for n in range(eval.shape[0]):
        value = eval[n, 3, 3, 4, :4]
        ax = plt.subplot()
        cm = [[value[0], value[2]], [value[3], value[1]]]
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax)
        plt.title('Accuracy', fontsize=15)
        plt.xlabel('False Positive Rate', fontsize=15)  # x-axis label with fontsize 15
        plt.ylabel('True Positive Rate', fontsize=15)  # y-axis label with fontsize 15
        path = "./Results/Confusion_%s.png" % (n + 1)
        # plt.savefig(path)
        plt.show()

def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'OOA', 'TFMVO', 'HGSO', 'CMBA', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(2):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' Statistical Report ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='OOA-MDDHN-AM')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='TFMVO-MDDHN-AM')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='HGSO-MDDHN-AM')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='CMBA-MDDHN-AM')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='RCMPA-MDDHN-AM')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['DBN', 'TCNN', 'RNN', 'RNN-TCNN', 'MDDHN-AM']
    for a in range(2):  # For 2 Datasets
        # Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)

        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_results_1():
    # matplotlib.use('TkAgg')
    eval = np.load('Evaluate_all.npy', allow_pickle=True)[:, :, :, :, :, 0]
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Algorithm = ['TERMS', 'OOA', 'TFMOA', 'HGSO', 'CMBA', 'PROPOSED']
    Classifier = ['TERMS', 'DBN', 'TCNN', 'RNN', 'RNN+TCNN', 'PROPOSED']
    for i in range(eval.shape[0]):
        value = eval[i, 0, 3, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - 75%-Algorithm Comparison ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - 75%-Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

        for j in range(len(Graph_Term)):
            # for i in range(eval.shape[0]):
            Graph = np.zeros((eval.shape[1], eval.shape[3]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[3]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, 4, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, 4, l, Graph_Term[j] + 4]

            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(4)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="DBN")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="TCNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="RNN")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="RNN-TCNN")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="MDDHN-AM")
            plt.xticks(X + 0.10, ('Deep\nFeature', 'TNSE\nFeature', 'Statistical\nFeature', 'Fused\nFeature'))
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                                 ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_bar.png" % (i+1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    plot_results_1()
    Plot_Confusion()
    plotConvResults()
    Plot_ROC_Curve()
