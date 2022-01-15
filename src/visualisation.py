from matplotlib import pyplot as plt


def draw_roc_curve(fpr, tpr, roc_score, algo_name):
    lw=2
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label=f"ROC curve (area = {roc_score:0.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.title(algo_name)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")
    plt.show()