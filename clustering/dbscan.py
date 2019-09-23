import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class dbscan():
    
    def __init__(self, x, y, eps=3, minpoints=4):
        self.df = pd.DataFrame({'x':x,'y':y})
        self.eps = eps
        self.minpoints = minpoints
        # New columns to dataframe
        self.df['setofpoints'] = 'unclassified'
        self.df['neighbors'] = 0
        self.df['neiborhood'] = None
        self.df['dataclass'] = 0
    
    
    def manhattan_dist(self, M, P):
        return np.abs(M[0]-P[0]) + np.abs(M[1]-P[1])
    
    
    def get_neighbors(self, i, data_class, reachable=False):
        print("Processing datapoint: ",i)

        neiborhood = []
        neighbors = 0
        update_class = False

        for x in range(len(self.df)):
            if i != x:
                # get M point
                M = [self.df.iloc[i]['x'], self.df.iloc[i]['y']]
                # get P point
                P = [self.df.iloc[x]['x'], self.df.iloc[x]['y']]
                # calculate the Manhattan distance from point M to P
                mdist = self.manhattan_dist(M,P)
                # Count the neiborhood
                if mdist <= self.eps:
                    neighbors += 1
                    neiborhood.append(x)

        self.df['neighbors'][i] = neighbors
        self.df['neiborhood'][i] = neiborhood
        
        # If don't have neighbors classify as noise
        if neighbors == 0:
            self.df['setofpoints'][i] = 'noise'
        # If don't have less neighbors than the configured on minpoints hyperparameter, classify as border
        if 0 < neighbors < self.minpoints:
            self.df['setofpoints'][i] = 'border'
            if reachable:
                self.df['dataclass'][i] = data_class
        # If don't have the number of neighbors iquals or higher than minpoints hyperparameter, classify as core
        if neighbors >= self.minpoints:
            self.df['setofpoints'][i] = 'core'
            self.df['dataclass'][i] = data_class
            update_class = True

            for k in neiborhood:
                if self.df['dataclass'][k] == 0:
                    update_class = self.get_neighbors(k, data_class, True)

        return update_class
    
    
    def run(self):
        # select a data point M to be analysed
        data_class = 1
        for i in range(len(self.df)):
            # Compare M to all other data points
            if self.df['setofpoints'][i] == 'unclassified':
                update_class = self.get_neighbors(i, data_class)

            if update_class:
                data_class += 1
                update_class = False
        
        return self.df


# Read the dataset
df = pd.read_csv("dataset.txt",  delimiter="\t")

# Plot the original data, with original classes
plt.scatter(df.x.values, df.y.values, c=df['oriclass'])
plt.savefig('original_classes.png')

# Run the clusterization
dbs = dbscan(df.x.values, df.y.values)
result = dbs.run()

# Plot the result
plt.scatter(result.x.values, result.y.values, c=result['dataclass'])
plt.savefig('classified_classes.png')