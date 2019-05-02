#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:24:33 2019

Purpose: This is a script to create the dash application for Activity detection. 

@author: zeski
"""

import os
import pickle

import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


import matplotlib.pyplot as plt


import plotly.graph_objs as go
import plotly.offline as pyo

df = pd.read_csv("components_dataframe.csv")


princomp_df = pd.DataFrame(columns = ["Compnent 1", "Component 2", "Component 3", "target"])






### The code for prediction
sample1 = df.iloc[:, :-1].sample().values




XGB_model = pickle.load(open("XGB_component_model.pickle", "rb"))
prediction_encoder = pickle.load(open("Activity_encoder.pickle", 'rb'))

prediction     = XGB_model.predict(sample1)
probabilities  = XGB_model.predict_proba(sample1)


### Dash Application code: 



app = dash.Dash()





app.layout = html.Div([
        ## Header 
        html.Div([
                html.P('Visualizing Activity!')
                
                
                
                ], style = {"textAlign" : "center", 
                                "color"     : "rgb(30,144,255)", 
                                "fontSize"  : 30}), 
        
       html.Hr(),

        ## Predicted Probabilities and Scatterplot
       html.Div(id ="prin_comp_vis", children = [
               
               
               dcc.Interval(id = "interval-component", 
                            interval =1750, 
                            n_intervals = 0
                            
                            ), 

              
              html.Div([dcc.Graph(id = "Probability of prediction1", 
                        
                        
                      
                      
                      )],style = {'display' : 'inline-block', "width":"100%", "paddingLeft" : 550}),
                
              html.Hr(),              
                            
              ## The scatterplot              
              html.Div([dcc.Graph(id = "updating_components_graph", 
                           
                           
                           
                           )
               
               
               
               
               ], style = { 'display' : 'inline-block', "width":"45%",  "paddingLeft" : 20}), 
        
        
              ## The updating counts of df
              
              html.Div( children = [
                      dcc.Graph(id = "updating_bar",
                                
                                
                                )
                      
                      
                      
                      ], style = { 'display' : 'inline-block', "width":"45%", "paddingLeft" : 20})
        
        ])


])


@app.callback([Output("Probability of prediction1", "figure"),
               Output('updating_components_graph', 'figure'), 
               Output("updating_bar", "figure")], 
                      
              [Input("interval-component", "n_intervals")]
              )
def update_probailites(n_intervals): 
        global df
        global XGB_model
        global prediction_encoder
        global princomp_df
            


        sample = df.sample(1)
        sample['target'] = prediction_encoder.inverse_transform(sample['target'])
        sample['predicted'] = prediction_encoder.inverse_transform(XGB_model.predict(sample.iloc[:, :-1].values))
        
        ## Necessary for first figure
        sample_probability = sample.iloc[:, :-2].values
        
        ## Necessary for second figure
        sample_plot = sample.loc[:, ["0", "1", "2", "predicted"]]
        sample_plot.columns = ["Compnent 1", "Component 2", "Component 3", "target"]
        princomp_df = pd.concat([princomp_df,sample_plot], axis = 0)
        
        
        ## Necessary for third figure
        bar_vis = pd.DataFrame(princomp_df.groupby("target").size()).reset_index()
        bar_vis.columns = ['Activity', "Counts"]

        
        
        
        
        probs = XGB_model.predict_proba(sample_probability)
        #print(probs)
        figure1 = {
                               'data' : [go.Bar(
                                       x = ['Laying', "Sitting", "Standing", "Walking", "Downstairs", "Upstairs"],#prediction_encoder.inverse_transform(list(df['target'].unique())), 
                                       y = probs.reshape(-1), 
                                       marker = {
                                               'color' : list(df['target'].unique()), 
                                               'colorscale' : 'Rainbow'
                                               }
                                       )], 
                                'layout' : go.Layout(
                                        title = "Probabilities of Class Predictions", 
                                        width = 700, 
                                        height = 350,
                                        xaxis = dict(
                                                title = "Activity", 
                                                
                                                
                                                ), 
                                        yaxis = dict(
                                                
                                                title = "Probability"
                                                )
                                        )
                               }
    
    
        figure2 = {
            'data' : [
                    go.Scatter3d(
                            x = princomp_df['Compnent 1'].values, 
                            y = princomp_df['Component 2'].values, 
                            z = princomp_df['Component 3'].values, 
                            mode = 'markers', 
                            marker = dict(

                                    size = 3,


                                    ),
                          text = ("Activity: " + princomp_df['target']) 
   
                            )
                    
                    
                    ], 
    
            'layout' : go.Layout(
                        title = "Seperation of Variables by 3 Top Principle Components", 
                        width = 900, 
                        height = 625,
                        scene = dict(
    
    
                        xaxis= dict(
                                title = "Princple Component 1"
                                ), 
                        yaxis=dict(
                                title = "Principle Component 2"
                                ), 
    
                        zaxis = dict(
                                title = "Principle Component 3"
                                )
                            )

                )
            } 
    
        figure3 = {
           'data'  :  [go.Bar(
                    x = bar_vis['Activity'], 
                    y = bar_vis['Counts'], 
                    marker = {
                            'color' : list(df['target'].unique()), 
                            'colorscale' : 'Jet'
                            }

                    ) ],
                   
           'layout' : go.Layout(
                   title = "Frequency Counts of Activities", 
                   height = 600,
                   xaxis = dict(
                           title = "Activty"
                           
                           ), 
                   yaxis = dict(
                           title = "Frequency Counts"
                           
                           )

                   )
            }
        
        return figure1, figure2, figure3


      
    

if __name__ == ("__main__"): 
    server = app.server
    app.run_server(debug = True)
    
   


