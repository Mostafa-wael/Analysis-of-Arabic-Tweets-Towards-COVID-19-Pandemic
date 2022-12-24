import pandas as pd
import nltk
# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo 
import plotly.graph_objects as go
import plotly.figure_factory as ff
import missingno as msno
from wordcloud import WordCloud
import random


def countPlot(data=None, x=None, y=None, palette='Reds_r', height=15, title=' ', subtitle=' ', description=' '):
    sns.set(style = 'whitegrid',
            rc = {'figure.figsize': (20,height)})
    g = sns.countplot(data=data, y=y, x=x, palette=palette)
    g.set_ylabel(' \n\n\n')
    g.set_xlabel(' \n\n\n\n\n')
    g.set_title(
        f'\n\n\n\n{title}\n\n'.upper(),
        loc = 'left',
        fontdict = dict(
            fontsize = 15,
            fontweight = 'bold'))
    g.set_yticklabels(
        [tick_label.get_text().title() for tick_label in g.get_yticklabels()],
        fontdict = dict(
            fontsize = 12.5,
            fontweight = 'medium'))
    plt.text(s = f'{description}',
             alpha = 0.5,
             x = 0,
             y = -.18,
             verticalalignment = 'baseline',
             horizontalalignment = 'left',
             transform = g.transAxes)
    g.bar_label(container = g.containers[0], padding = 10,)
    plt.text(s = ' ', x = 1.08, y = 1, transform = g.transAxes)
    sns.despine()
    return g

def piePlot(data=None, value='Percentage', name='Sentiment', title=' ', subtitle=' ', description=' '):
    data_pie = pd.DataFrame(data.value_counts() / data.shape[0]*100).reset_index()
    data_pie.columns = [name, value]
    fig = px.pie(data_pie, values=value, names=name, title=title)
    fig.update_layout(
    title=title, title_x=0.48)
    fig.show()


def Freq_df(sentence):
    words = [item for sublist in sentence for item in sublist]
    Freq_dist_nltk = nltk.FreqDist(words)
    df_freq = pd.DataFrame.from_dict(Freq_dist_nltk, orient='index')
    df_freq.columns = ['Frequency']
    df_freq.index.name = 'Term'
    df_freq = df_freq.sort_values(by=['Frequency'],ascending=False)
    df_freq = df_freq.reset_index()
    return df_freq

def plotDF(freq_df, x, y, title="", xlab="", ylab="", dpi=100):
    top_10 = freq_df[:10]
    fig = px.bar(top_10, x = 'Term', y = 'Frequency',text = 'Frequency', color='Term',
                color_discrete_sequence=px.colors.sequential.PuBuGn, title = 'Rank of terms')
    for idx in range(len(top_10)):
        fig.data[idx].marker.line.width = 2
        fig.data[idx].marker.line.color = "black"
    fig.update_traces(textposition='inside',
                    textfont_size=11)
    fig.show()
