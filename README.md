# Plotting the Optimal Distance for Data Scientists in Python using the K-Nearest Neighbour K-NN Algorithm

This example uses the K-Nearest Neighbour K-NN to connect several points and plotting those as a graph on a Plotly map. This is a quick and simple example that data scientists can use to illustrate a path(s) for their professional or academic papers.

A brief file structure overview of the repository is provided. The knn_map.py is in the root directory. The data folder houses a list of target geocoordinates in a csv file. The out map is generated in the output folder.

    /
    knn_map.py

    - / data /
    geocoordinates.csv

    - / output /
    knn_map.jpg
  
The geocoordinates.csv includes the following target geocoordinates.

    LATITUDE	LONGITUDE
    48.13485	11.5173913
    48.1348182	11.577103
    48.1492002	11.5592469
    48.11005	11.59344
    48.158903	11.5856
  
Before jumping into the code the following requirements and packages are needed to run the code:

    Python 3.10.6
    pip3 install numpy
    pip3 install scipy
    pip3 install pandas
    pip3 install plotly
    pip3 install kaleido

First the packages that were just installed are imported into our file knn_map.py

    import numpy as np
    from numpy import sqrt 
    from scipy.spatial.distance import pdist, squareform
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

Next a function generate_knn_grid is created which returns the K-NN grid. This function takes the dataset of geocoordinates in the data folder as a parameter.

    def generate_knn_grid(df): 

Inside the function the parameters for the KNN algorithm are defined. N is the number of nearest neighbours or nearest nodes we want to connect. As a standard practice K is set to the square root of the number of nearest neighbours. X stacks the geocoordinate data point into a 2D Array.

            # Parameters for the k-nearest neighbor algorithm
            N = len(list(df.LATITUDE))
            k = round(sqrt(N))
            X = np.column_stack((lat, lon))

Next the Euclidean distances between the geocoordinates or nodes can be generated. pdist returns a matrix of distance between each pairs of nodes.

            # matrix of pairwise Euclidean distances will determine which dector node 
            is connected to another dector node
            distmat = squareform(pdist(X, 'euclidean'))

For each node the nearest neighbours are determined using the Euclidean distance matrix.

            # select the kNN for each datapoint
            neighbors = np.sort(np.argsort(distmat, axis=1)[:, 0:k])

Using the this information the geocoordinates belong to connectors or edges between the nodes can be calculated.

            # get edge coordinates for the lines connecting the nodes
            coordinates = np.zeros((N, k, 2, 2))

            for i in np.arange(len(list(df.LATITUDE))):
                for j in np.arange(k):
                    coordinates[i, j, :, 0] = np.array([X[i,:][1], X[neighbors[i, j], :][1]])
                    coordinates[i, j, :, 1] = np.array([X[i,:][0], X[neighbors[i, j], :][0]])

The calculated information is then returned along with N and k.

            return N, k, coordinates

Next the function for plotting the map is created. It takes the returned information from the pervious function as parameters.

      def plot_map(N, k, coordinates):

The map figure is created and styled using Plotly. An API access token from Mapbox can be acquired for free after signing up. Mapbox has many beautiful maps from a variety of contributors. These look nice in academic papers. Remember to attribute the map creator. Times New Roman was used and the shape was set to a square. The nodes are added from the dataset.

            fig = px.scatter_mapbox(
                df, 
                lat="LATITUDE", 
                lon="LONGITUDE", 
                zoom=12.2,
                color_discrete_sequence=["rgb(255, 203, 3)"],
                title="<span style='font-size: 32px;'><b>K-Nearest Neighbor KNN
                Map</b></span>",
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
                            legend=dict(yanchor="top", y=1, xanchor="left",x=0.9),
                            title=dict(yanchor="top", y=.85, xanchor="left",x=0.085),
                            font_family="Times New Roman",
                            font_color="#333333",
                            title_font_size = 32,
                            font_size = 18)

The node diameters are set to 15.

            # diameter of the plots 
            fig.update_traces(marker={'size': 15})

Using the information from the parameters the connectors are generated and added to the map.

            # add line connectors
            lines = coordinates.reshape((N*k, 2, 2))
            
            i = 0
            for row in lines:
                fig.add_trace(go.Scattermapbox(lon=[lines[i][0].[0],
                    lines[i][1][0]], 
                    lat=[lines[i][0][1],
                    lines[i][1][1]], 
                    mode='lines', 
                    showlegend = False, 
                    line=dict(color='#ffcb03')))
                i += 1

The line connectors are z-indexed below the nodes.

            # line connectors layered below plots    
            fig.data = fig.data[::-1]  

Finally the map is saved in the output folder and rendered in the browser.

            # Save map in output folder
            print("Saving image to output folder...");
            fig.write_image('output/knn_map.jpg', scale=5)
            
            # Show map in web browser
            print("Generating map in browser...");
            fig.show()

After defining the functions we import our data from the data directory and format it.

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

Finally we call our two functions to compute and create our KNN map.

    N, k, coordinates = generate_knn_grid(df)

    plot_map(N, k, coordinates)

KNN could be used, for instance, to predict how viruses and other pathogens will move across a given region or spatial area by creating a graph from the individual cases. However for traffic-related forecasting, it is preferable to construct the graph directly from the road network using the Dijkstra algorithm.

Sources for KNN: https://stackoverflow.com/questions/50040310/efficient-way-to-connect-the-k-nearest-neighbors-in-a-scatterplot-using-matplotl
