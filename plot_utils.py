from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import median
from scipy.stats import linregress
from collections import defaultdict
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import pandas as pd

############## MAIN PLOTTING FUNCTIONS ###############################

def plot_freq_dist(cnt, title=None, x_axis='Entity mentions', loglog=False, b=2, save=False):
	"""
	Plot a frequency distribution from a dictionary where keys are strings, and values are their frequencies.
	"""
	fig = plt.figure()

	y = OrderedDict(cnt.most_common())
	v=list(y.values())
	k=np.arange(0,len(v),1)
	if loglog:
		plt.loglog(k,v, basex=b)
	else:
		plt.plot(k,v)
	plt.ylabel('Frequency')
	plt.xlabel(x_axis)
	if title:
		if loglog:
			p_title = 'Distribution of %s (log-log)' % title
		else:
			p_title = 'Distribution of %s' % title
		plt.title(p_title)
	plt.show()
	if save:
		if title:		
			fig.savefig('img/%s.png' % p_title.lower().replace(' ', '_'), bbox_inches='tight')
		else:
			fig.savefig('img/%d.png' % random.randint(0,1000000), bbox_inches='tight')	

def plot_multi_freq_dist(forms_cnt, kind='', x_axis='Entity mentions', loglog=False, b=2, save=False):
    """
    Plot multiple frequency distributions from multiple dictionaries where keys are strings, and values are their frequencies.
    """
    fig = plt.figure()
    cnt=0
    scales=['0.1', '0.5']
    for title, data_forms_cnt in forms_cnt.items():
        gray_level=scales[cnt]
        print(title)
        y = OrderedDict(data_forms_cnt.most_common())
        v=list(y.values())
        k=np.arange(0,len(v),1)
        if loglog:
                plt.loglog(k,v, basex=b, label=title.upper(), color=gray_level)
        else:
                plt.plot(k,v, color=gray_level, label=title.upper())
        cnt+=1
    plt.legend(loc='upper right', frameon=1)
    plt.ylabel('FREQUENCY')
    plt.xlabel(x_axis)
    if kind:
        if loglog:
                p_title = 'Log-Log distribution of %s' % kind
        else:
                p_title = 'Distribution of %s' % kind
        plt.title(p_title)
    plt.show()
    if save:
            if title:
                    fig.savefig('img/%s.png' % p_title.lower().replace(' ', '_'), bbox_inches='tight')
            else:
                    fig.savefig('img/%d.png' % random.randint(0,1000000), bbox_inches='tight')


def plot_line_with_whiskers(x, y, xl='', yl='', title='', estimators=['mean', 'median'], xlim=None, show_title=True, save=False):
    """
    Plot a line with whiskers, showing a standard deviation from the estimator.
    """
    for est in estimators:
        fig = plt.figure()

        amb_data = {xl: x, 
                    yl: y}

        df = pd.DataFrame(amb_data)

        if est=='mean': # mean
            ax = sns.pointplot(x=xl, y=yl, data=df)
        else: # median
            ax = sns.pointplot(x=xl, y=yl, data=df, estimator=median)

        if title:
            plt_title='%s (estimator=%s)' % (title, est)
            if show_title:
                plt.title(plt_title)

        if xlim:
                ax.set(xlim=(xlim[0], xlim[1]))
        ax.set(ylim=(0, None))

        plt.show()

        if save:
            if title:
                fig.savefig('img/%s.png' % plt_title)
            else:
                fig.savefig('img/%d.png' % random.randint(0,1000000))

def multi_plot_line_with_whiskers(x, y, xl='', yl='', a=None, xlim=None, save=False, system=''):
    """
    Plot multiple lines on the same line plot figure with whiskers.
    """
    amb_data = {xl: x,
                yl: y}

    df = pd.DataFrame(amb_data)

    ax = sns.pointplot(x=xl, y=yl, data=df, ax=a)
    ax.set_title(system.upper())

    if xlim:
            ax.set(xlim=(xlim[0], xlim[1]))
    ax.set(ylim=(0, 100.0))

def plot_scores(scores, title=''):
	"""
	Plot multiple bar plots, one per system. Each plot shows the system performance across different evaluation categories.
	"""
	dpoints = np.array(scores)

	plt.gray()
	fig = plt.figure()
	ax = fig.add_subplot(111)

	space = 0.4

	evals = np.unique(dpoints[:,0])
	systems = np.unique(dpoints[:,1])

	systems=[s.upper() for s in systems]

	evals=['overall', 'ambiguous forms'] #, 'forms with nils & non-nils' ]
	colors=['#222222', '#aaaaaa', '#666666']
	print(evals)
	print(systems)

	n = len(evals)

	width = (1 - space) / (len(evals))
	print("width:", width)

	indeces = range(1, len(systems)+1)

	for i,cond in enumerate(evals):
		print("evaluation:", cond)

		vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)
		pos = [j - (1 - space) / 2. + i * width + width/len(evals) for j in range(1,len(systems)+1)]
		br = ax.bar(pos, vals, width=width, label=cond, color=colors[i], align = 'center') #cm.Accent(float(i) / n))
		autolabel(br, ax)
	    
	ax.set_xticks(indeces)
	ax.set_xticklabels(systems)
	plt.setp(plt.xticks()[1], rotation=0)    

	ax.set_ylim(ymax=1.1)
	ax.set_ylabel("F1-SCORE")

	handles, labels = ax.get_legend_handles_labels()
	legend = ax.legend(handles[::1], labels[::1], loc='upper center', bbox_to_anchor=(0.22, 1.02),  shadow=True, ncol=1, frameon=1)
	frame = legend.get_frame()
	frame.set_edgecolor('gray')
	plt.show()

	if title:
		fig.savefig('img/%s.png' % title.lower().replace(' ', '_'), bbox_inches='tight')
	else:
		fig.savefig('img/%d.png' % random.randint(0,1000000), bbox_inches='tight')

def plot_prf(data, systems, a, maxrank=12, title=''):
    """
    Plot precision, recall, and F1 for different ranks.
    """
    a_list=[]
    for rank in range(1,maxrank+1):
        s=0
        for system in systems:
            s+=data[system][rank]
        a_list.append(s/len(systems))
    print(np.arange(1,maxrank+1))
    a.plot(np.arange(1,maxrank+1), a_list, 'b-o')
    a.set_xlabel("RANK")
    a.set_title(title)
    a.xaxis.set_ticks(np.arange(1,maxrank+1))

################### MAIN PLOTTING FUNCTIONS DONE ###############################
################### HELPER FUNCTIONS ###########################################


def autolabelh(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        width = rect.get_width()
        ax.text(1.05*width, rect.get_y() + rect.get_height()/2.,
                int(width),
               va='center')

def autolabel(rects, ax):
	"""
	Attach a text label above each bar displaying its height
	"""
	for rect in rects:
		height = rect.get_height()
		print(height)
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
			round(height,2), fontsize=10,
			ha='center', va='bottom')

###### HELPER FUNCTIONS DONE ###########
###### DEPRECATED ######################

def lmplot(dist1, dist2):
    """
    Plot correlation between two variables with LM plot, to estimate the shape of the distribution.
    """
    data={
        'form frequency': dist1,
        'accuracy': dist2
    }
    df=pd.DataFrame(data)
    ax = sns.lmplot(data=df, x='form frequency', y='accuracy', lowess=True)


def frequency_correlation(freq_dist, other_dist, min_frequency=0, title=None, x_label='', y_label='', xlim=None, save=False):
	"""
	Plot a frequency correlation between two valiables, both given with dictionaries, that share keys.
	"""
	fig = plt.figure()


	other_per_frequency = defaultdict(int)
	count_per_frequency = defaultdict(int)
	for form,frequency in freq_dist.items():
		if frequency>min_frequency:
			if form in other_dist:
				count_per_frequency[frequency]+=1
				other_per_frequency[frequency]+=other_dist[form]

	x=[]
	y=[]
	for frequency in sorted(count_per_frequency):
#    		print(frequency, other_per_frequency[frequency]/count_per_frequency[frequency])
    		x.append(frequency)
    		y.append(other_per_frequency[frequency]/count_per_frequency[frequency])
	plt.plot(x,y, marker='o')
	plt.ylabel(y_label)
	plt.xlabel(x_label)
	if xlim:
		plt.xlim(xlim[0], xlim[1])
	plt.ylim(ymin=0)
	if title:
		plt.title(title)
	plt.show()

	if save:
		if title:
			fig.savefig('img/%s.png' % title.lower().replace(' ', '_'), bbox_inches='tight')
		else:
			fig.savefig('img/%d.png' % random.randint(0,1000000), bbox_inches='tight')


def annotated_heatmap(x_labels, y_labels, values, x_lbl='levels'):
    """
    Draw a heat map from three lists, one with X labels, one with Y labels, and one with the values to be shown.
    """
    x=[]
    y=[]
    vals=[]
    i=0
    for x_val in x_labels:
        for y_val in y_labels:
            x.append(x_val)
            y.append(y_val)
            vals.append(values[i])
            i+=1
    print(x, y, vals)

    data={}
    data['dataset']=x
    data[x_lbl]=y
    data['values']=vals
    
    fig, ax = plt.subplots(figsize=(len(y_labels)+3,0.75*len(x_labels)))         # Sample figsize in inches
    
    df = pd.DataFrame.from_dict(data)
    
    result = df.pivot(index='dataset', columns=x_lbl, values='values')

    print(result)
    ax = sns.heatmap(data=result, annot=True, fmt="d", cmap='cubehelix', )
    
    fig.savefig('img/%s.png' % x_lbl)

#####################################################################
