'''
Originally posted by Paul Broderson on Stackoverflow
https://stackoverflow.com/questions/50040310/efficient-way-to-connect-the-k-nearest-neighbors-in-a-scatterplot-using-matplotl
K-Nearest-Neighbor KNN the simplest machine learning algorithm 
Using Bahnhofplatz Nord as an example to connect the sensors
'''

import numpy as np
from numpy import sqrt 
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

    
##### Create a grid of the nearest neighbors
def generate_knn_grid(df): 

    # Parameters for the k-nearest neighbor algorithm
    N = len(list(df.LATITUDE))
    k = round(sqrt(N))
    X = np.column_stack((lat, lon))

    # matrix of pairwise Euclidean distances will determine which dector node is connected to another dector node
    distmat = squareform(pdist(X, 'euclidean'))

    # select the kNN for each datapoint
    neighbors = np.sort(np.argsort(distmat, axis=1)[:, 0:k])

    # get connector coordinates for the lines connecting the nodes
    coordinates = np.zeros((N, k, 2, 2))

    for i in np.arange(len(list(df.LATITUDE))):
        for j in np.arange(k):
            coordinates[i, j, :, 0] = np.array([X[i,:][1], X[neighbors[i, j], :][1]])
            coordinates[i, j, :, 1] = np.array([X[i,:][0], X[neighbors[i, j], :][0]])

    return N, k, coordinates

def plot_map(N, k, coordinates):


    fig = px.scatter_mapbox(
        df, 
        lat="LATITUDE", 
        lon="LONGITUDE", 
        zoom=12.2,
        color_discrete_sequence=["rgb(255, 203, 3)"],
        title="<span style='font-size: 32px;'><b>K-Nearest Neighbor KNN Map</b></span>",
        opacity=.8,
        width=1000,
        height=1000,
        center=go.layout.mapbox.Center(
                lat=48.14,
                lon=11.57,
            ),
        size_max=15
        )

    # Now using Mapbox
    fig.update_layout(mapbox_style="light", 
                    mapbox_accesstoken="",
                    legend=dict(yanchor="top", y=1, xanchor="left", x=0.9),
                    title=dict(yanchor="top", y=.85, xanchor="left", x=0.085),
                    font_family="Times New Roman",
                    font_color="#333333",
                    title_font_size = 32,
                    font_size = 18)

    # diameter of the plots 
    fig.update_traces(marker={'size': 15})

    # add line connectors
    lines = coordinates.reshape((N*k, 2, 2))
    i = 0
    for row in lines:
        fig.add_trace(go.Scattermapbox(lon=[lines[i][0][0],lines[i][1][0]], lat=[lines[i][0][1],lines[i][1][1]], mode='lines', showlegend = False, line=dict(color='#ffcb03')))
        i += 1
        
    # line connectors layered below plots    
    fig.data = fig.data[::-1]

    # Save map in output folder
    print("Saving image to output folder...");
    fig.write_image('output/knn_map.jpg', scale=5)

    # Show map in web browser
    print("Generating map in browser...");
    fig.show()


# Data Import Path
SENSORS_CSV   = 'data/geocoordinates.csv'

# Data Import Path
df = pd.read_csv(SENSORS_CSV)


# Keep only relevant columns
df = df.loc[:, ("LATITUDE", "LONGITUDE")]

# Remove missing geocoordinates
df = df[(df["LATITUDE"] != "NEIN") & (df["LONGITUDE"] != "NEIN")]

# Format the latitudes and longitudes
lon = []; lat = [];

for row in df["LATITUDE"]:
    lat.append(float(row))
    
for row in df["LONGITUDE"]:
    lon.append(float(row))

N, k, coordinates = generate_knn_grid(df)

plot_map(N, k, coordinates)