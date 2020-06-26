# Angelo Salton <gsalton4@hotmail.com>

import pandas as pd
import pickle
from pycaret import anomaly, arules, regression, classification, clustering, nlp
import streamlit as st

# get dataset options (pycaret examples)
# pc_dfs = pd.read_html('https://pycaret.org/get-data/', header=0)[0]
# pickle.dump(pc_dfs, open('assets/pycaret_datasets.pkl', 'wb'))
pc_dfs = pickle.load(open('assets/pycaret_datasets.pkl', 'rb'))

st.title('PyCaret Explorer')
st.write('Hello!')

opt_task = pc_dfs['Default Task'].unique().tolist()
optsel_task = st.sidebar.selectbox('Select a task:', opt_task)

opt_dataset = pc_dfs['Dataset'][pc_dfs['Default Task']==optsel_task].unique().tolist()
optsel_dataset = st.sidebar.selectbox('Select a dataset:', opt_dataset)

if st.sidebar.button('Run'):
    # Download data
    df = pd.read_csv(f'https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/{optsel_dataset}.csv')

    df = df.copy().sample(200)

    # Get target
    df_target = pc_dfs.loc[pc_dfs['Dataset']==optsel_dataset,'Target Variable'].tolist()[0]

    # Get task
    df_task = pc_dfs.loc[pc_dfs['Dataset']==optsel_dataset,'Default Task'].tolist()[0]
    
    # Describe data
    st.write(f'This dataset has {df.shape[0]} samples and {df.shape[1]} features. Target variable is {df_target}.')
    st.dataframe(df.head())

    if df_task in ['NLP / Regression','Regression']:

        # Setup PyCaret
        with st.spinner('PyCaret setup is running...'):
            pycset = regression.setup(data=df, target=df_target)

        # Compare models
        st.dataframe(regression.compare_models())

        # End
        st.success('End of execution!')

    if df_task in ['Classification (Binary)','Classification (Multiclass)']:

        # Setup PyCaret
        with st.spinner('PyCaret setup is running...'):
            pycset = classification.setup(data=df, target=df_target)

        # Compare models
        st.dataframe(classification.compare_models())

        # End
        st.success('End of execution!')

    if df_task in ['NLP']:

        # Setup PyCaret
        with st.spinner('PyCaret setup is running...'):
            pycset = nlp.setup(data=df, target=df_target)

        # Compare models
        #st.dataframe(classification.compare_models())

        # End
        st.success('End of execution!')

    if df_task in ['Association Rule Mining']:

        # Setup PyCaret
        with st.spinner('PyCaret setup is running...'):
            pycset = arules.setup(data=df, target=df_target)
        
        # Compare models
        #st.dataframe(arules.compare_models())

    if df_task in ['Anomaly Detection']:

        # Setup PyCaret
        with st.spinner('PyCaret setup is running...'):
            pycset = anomaly.setup(data=df, target=df_target)

        # Compare models
        #st.dataframe(anomaly.compare_models())

    else:
        st.error('Method not implemented.')
