import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

plt.rcParams['figure.facecolor'] = '#1A202C'
plt.rcParams['axes.facecolor'] = '#1A202C'  # Couleur de fond des axes
plt.rcParams['axes.edgecolor'] = '#FFFFFF'  # Couleur des bordures des axes
plt.rcParams['axes.labelcolor'] = '#FFFFFF'  # Couleur des étiquettes des axes
plt.rcParams['xtick.color'] = '#FFFFFF'  # Couleur des ticks sur l'axe des x
plt.rcParams['ytick.color'] = '#FFFFFF'  # Couleur des ticks sur l'axe des y
plt.rcParams['text.color'] = '#FFFFFF'

def scat2cat(data:DataFrame, regLine=None, title=None, legend=False, lim0=False, sp=10, xlabel=None, ylabel=None, autoName=False, file=None):

    x_cat1 = data[data.iloc[:, 2] == 0].iloc[:,0]
    y_cat1 = data[data.iloc[:, 2] == 0].iloc[:,1]
    x_cat2 = data[data.iloc[:, 2] == 1].iloc[:,0]
    y_cat2 = data[data.iloc[:, 2] == 1].iloc[:,1]

    # plt.ion()   # Continue d'excuter le code durant le plt.show()
    plt.scatter(x_cat1, y_cat1, color='#3498db', s=sp)
    plt.scatter(x_cat2, y_cat2, color='#ff5733', s=sp)

    if regLine:
        plt.plot(regLine[0], regLine[1])

    if lim0:
        plt.xlim(0, )
        plt.ylim(0, )

    if autoName:
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        plt.title(data.columns[2])
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if legend:
        plt.legend()

    if file:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()
