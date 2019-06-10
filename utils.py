import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import matplotlib.pyplot as plt

#from plotly.offline import init_notebook_mode, iplot
#from plotly.graph_objs import Contours, Histogram2dContour, Marker, Scatter 
#from plotly import tools

import cufflinks as cf 
cf.go_offline()
#from collections import Counter

def identify_data_domain(data, feature):
     
    #data domain list
    dd_list = []
    
    #flat data domain list
    flat_dd_list = []
    
    #scanning the data frame  #total of different traits   
    for i in data.index:
       # breaking down the feature in search of data domain
        word = str(data.loc[i, feature]).split(",")
        # adding the data domain in a lista
        dd_list.append(word)        
             
    #scanning the dd_list in searching for the data domains lists in order to breaking down 
    for sublist in dd_list:
        for item in sublist:
            flat_dd_list.append(item)   
               
    return pd.Series(flat_dd_list).str.lstrip().drop_duplicates().tolist()

    
def create_columns (data, new_columns):
    #creating the new columns
    for idx, item in enumerate(new_columns):
        #print(new_columns[idx])
        #setting 0 as default value for the new rows of the new columns
        data[new_columns[idx]] = 0    
    return data


def hot_encoding(data, feature):
    
    new_columns = identify_data_domain(data, feature)
    
   # feature = str(feature)
    
    new_data = create_columns(data, new_columns)
    
    #
    print("Quantidade de novas colunas: " + str(len(new_columns)))
   
    #
   # for index, item in enumerate(new_columns):    
       # print("index: " + str(index) + "valor: " + new_columns[index])                
        
    for ids, row in new_data.iterrows():
        #print("***************************************************************")
        #print(str(ids))
        #print(new_data.loc[[ids]][feature])
        #print("new-dataat =" + srt(new_data.at[ids, feature]))                     
        #print("iterando o dataset indice:" + str(ids))# + " " +  new_data.at[ids, feature])
        for ic, item in enumerate(new_columns):
            #if ic == ic:
                #print("ic =" + str(ic) + "feature = " + feature + "new-dataat =" + new_data.at[ids, feature] + "new_columns[ic] ="
                #      + new_columns[ic])
                #var =  new_data.at[ids, feature].find(new_columns[ic])
                #print("var = " + str(var))
                #print("iterando as colunas:" + str(ic) + " " + new_columns[ic])
                #print("resultado do if: " +str( new_data.at[ids, feature].find(new_columns[ic]) >= 0))
                #print("resultado: " +str( new_data.at[ids, feature].find(new_columns[ic])))
            if new_data.at[ids, feature].find(new_columns[ic]) >= 0:            
                new_data.at[ids, new_columns[ic]] = 1
                 #   print("valor em trait " +  new_data.at[ids, feature] + " == " + " valor da coluna:" + new_columns[ic])
                 #   print("atribui 1")
                 #   print("##################################################################")
                    #data_new.at[i, new_columns[idx]] = 1'''
        
    to_drop = [feature]
    new_data.drop(new_data[to_drop], axis=1, inplace=True)
    return new_data


def plot_missing(df):
    colunas = df.columns
    percent_val_faltantes = (((df.isnull().sum() * 100 / len(df))))
    missing_value_df = pd.DataFrame({    'Variável': colunas, '% de valores faltantes': ((percent_val_faltantes))
                                    })

    missing_value_df = (missing_value_df[(missing_value_df['% de valores faltantes'] > 0)])

    missing_value_df.reset_index(drop=True, inplace=True)

    val = missing_value_df['% de valores faltantes']
    missing_value_df['% de valores faltantes'] = missing_value_df['% de valores faltantes'].round(2)
    missing_value_df = missing_value_df.sort_values('% de valores faltantes', ascending=True)
    
    #PLOT MISSING VALUES
    val = list(missing_value_df['% de valores faltantes'])
    var = list(missing_value_df['Variável'])

    x = val
    y = var

    x_text = []
    y_text = []

    for i in x:
        i = str(i)+'%'
        x_text.append(i)
  
    widthpadrao=0
    heightpadrao=0
    
    if len(y) >= 40:
        widthpadrao+=1200
        heightpadrao+=6000
        l=300
        b=620
        t=300
        r=0
    elif len(y) >= 33 and len(y) < 40:
        widthpadrao+=1200
        heightpadrao+=len(y)*45
        l=300
        b=620
        t=300
        r=0
    elif len(y) >= 10 and len(y) < 33:
        widthpadrao+=len(y)*35
        heightpadrao+=len(y)*30
        l=200
        b=65
        t=70
        r=0
    elif len(y) >= 5 and len(y) < 10:
        widthpadrao+=len(y)*70
        heightpadrao+=len(y)*85
        l=100
        b=35
        t=45
        r=0
    elif len(y) >= 2 and len(y) < 5:
        widthpadrao+=len(y)*150
        heightpadrao+=len(y)*120
        l=90
        b=30
        t=35
        r=0
    else:
        widthpadrao+=400
        heightpadrao+=410
        l=70
        b=80
        t=30
        r=0        
        
    data = [go.Bar(
        x=x,
        y=y,
        text=x_text,
        textposition = 'auto',
        orientation = 'h',
        width = 0.9
    )
           ]

    layout = go.Layout(
        title= 'Percentual of missing Data (%)',
        autosize=True,
        width=widthpadrao,  
        height=heightpadrao,                            
        margin=dict(l=l,r=r,b=b,t=t, pad=0)
    )
    
    fig = go.Figure(data=data, layout=layout)
    #iplot(data, filename='horizontal-bar')
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='horizontal-bar')    
    
def thanos(features_thanos, seed, feature, rate):
    np.random.seed(seed)
    mask = np.random.choice([True, False], size=features_thanos[feature].shape)
    print(mask)
    target = features_thanos[feature].mask(np.random.choice([True, False], size=features_thanos[feature].shape, p=[rate,(1-rate)]))
    features_thanos[feature] = target
    features_thanos.dropna(how='all', subset=[feature], inplace = True) 
    return features_thanos
    
def highCorrelated_drop(data):
    # Create correlation matrix
    corr = data.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.show()
    
    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.90
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    print("Variables to be droped:", to_drop)
    
    data.drop(data[to_drop], axis=1, inplace=True)
    return data 

def features_to_drop(data, dont_drop):
    dont_drop_list = dont_drop.split(",")
    #dont_drop_list
    dont_drop = dont_drop.replace("' ", "'").strip()
    #dont_drop
    #print(len(dont_drop_list))
    #features_final.columns
    #print(len(features_final.columns))
    columns = data.columns
    #print(len(columns))
    to_drop = []

    for ids, item in enumerate(columns):
        #print("Item na coluna:", item)
        #print(dont_drop)
        #print("Lista para dropar:" ,to_drop)
        #print(dont_drop.find(item) < 0)
        if dont_drop.strip().find(item.strip()) < 0: 
        #if dont_drop_list[.string.find(item) <= 0:
            #print(item)
            to_drop.append(item)                

        
    #print(to_drop)
    
    #print(len(to_drop))
    #print(len(features_final.columns))
    selected_features = pd.DataFrame(data=data)
    selected_features.shape
    #print(to_drop)
    selected_features.drop(selected_features[to_drop], axis=1, inplace=True)
    selected_features.shape
    selected_features.head(1)
    return selected_features
    
   
    