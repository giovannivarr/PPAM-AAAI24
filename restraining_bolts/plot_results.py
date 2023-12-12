#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import argparse


def loaddata(filename, datafiles=False):

    try:
        if datafiles:
            fname = str(filename).replace('.dat','')
            fname = fname.replace('data/','')
            a = np.loadtxt('data/' + fname + '.dat', delimiter=",")
        else:
            a = np.loadtxt(filename, delimiter=',')
    except:
        print("Error in loading file")
        return None, None, None

    try:
        it = np.array(a[:,0])  # iteration vector
        tm = np.array(a[:,1])  # time vector
        sv = a[:,2]  # score vector
        rv = a[:,3]  # reward vector
        gv = a[:,4]  # goal reached vector
#        ov = a[:,6]  # optimal (no exploration) vector
    except: # old version
        sv = a[:,0]  # score vector
        rv = a[:,1]  # reward vector
        gv = a[:,2]  # goal reached vector
        tm = range(0,len(rv))
#        ov = a[:,4]  # optimal (no exploration) vector

    # sv for scores, rv for rewards
    return it, rv, filename



def getplotdata(tm,data):
    x = [] # x axis vector
    y = [] # y axis vector 
    ytop = []  # confidence interval (top-edge)
    ybot = []  # confidence interval (bottom-edge)

    n = len(data)
    d = 100 # size of interval

    for i in range(0,int(n/d)):
        di = data[i*d:min(n,(i+1)*d)]
        ti = tm[i*d:min(n,(i+1)*d)]
        if (len(ti)>0):
            x.append(np.mean(ti)/d)

            #for score
            #y.append(np.rint(np.mean(di)))

            #for reward
            y.append(np.mean(di))

            # This is from the original RB code,
            # now instead we take the 25th and 75th percentile across all different runs,
            # and for individual runs we just plot the averages
            #ytop.append(np.mean(di)+0.5*np.std(di))
            #ybot.append(np.mean(di)-0.5*np.std(di))

    #return x,y,ytop,ybot
    return x,y


def showplots(xx,yy,yytop,yybot,yylabel,save):

    colors = ['r','b','g','cyan','yellow','magenta']
    
    ytop = max(max(l) for l in yytop)

    plt.ylim(bottom = 0, top = ytop*1.2)
    plt.title("")
    plt.xlabel('Interval (Iteration/100)')

    #score
    #plt.ylabel('Score')

    #reward
    plt.ylabel('Median Reward')

    for i in range(0,len(xx)):
        # comment next line when plotting score
        plt.fill_between(xx[i], yytop[i], yybot[i], facecolor=colors[i], alpha=0.25)
        plt.plot(xx[i],yy[i],colors[i],label=yylabel[i])

    #plt.xticks(np.arange(0, 101, 10))
    plt.legend()
    plt.grid()

    if save is not None:
        plt.savefig(save)
        print('File saved: ',save)

    plt.show()


def plotdata(datafiles = None, save = None):
    if not datafiles:
        xx_ppam, xx_rb = [], []
        yy_ppam, yy_rb = [], []
        yy_label = ['RB', 'PPAM']

        for file in os.listdir('./data/PPAM/'):
            filename = os.fsdecode(file)
            if filename.endswith('.dat'):
                tm, rv, fname = loaddata('./data/PPAM/' + filename)
                x,y = getplotdata(tm, rv)
                if len(xx_ppam) == 0:
                    xx_ppam = x
                yy_ppam += [y]

        for file in os.listdir('./data/RB/'):
            filename = os.fsdecode(file)
            if filename.endswith('.dat'):
                tm, rv, fname = loaddata('./data/RB/' + filename)
                x,y = getplotdata(tm, rv)
                if len(xx_rb) == 0:
                    xx_rb = x
                yy_rb += [y]

        yy_ppam_25 = np.percentile(yy_ppam, 25, axis=0)
        yy_ppam_median = np.median(yy_ppam, axis=0)
        yy_ppam_75 = np.percentile(yy_ppam, 75, axis=0)
        yy_rb_25 = np.percentile(yy_rb, 25, axis=0)
        yy_rb_median = np.median(yy_rb, axis=0)
        yy_rb_75 = np.percentile(yy_rb, 75, axis=0)
        showplots([xx_rb, xx_ppam], [yy_rb_median, yy_ppam_median], [yy_rb_75, yy_ppam_75], [yy_rb_25, yy_ppam_25], yy_label, save)

    else:
        xx, yy, yybot, yytop, yylabel = [], [], [], [], []
        for f in datafiles:
            tm,rv,fname = loaddata(f, True)
            if tm is not None:
                #x,y,ytop,ybot = getplotdata(tm,rv)
                x,y = getplotdata(tm, rv)
                xx += [x]
                yy += [y]
                #yytop += [ytop]
                #yybot += [ybot]
                yylabel += [fname]

    yylabel = ['RB', 'PPAM']
    if (len(xx)>0):
        showplots(xx,yy,yy,yy,yylabel,save)



# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot results')
    #parser.add_argument('file', type=str, help='File name with data')
    #parser.add_argument('--reward', help='plot reward', action='store_true')
    #parser.add_argument('--score', help='plot score', action='store_true')
    parser.add_argument('-save', type=str, help='save figure on specified file', default=None)

    parser.add_argument('-datafiles', nargs='+', help='Data files to plot')

    args = parser.parse_args()

    plotdata(args.datafiles, args.save)


