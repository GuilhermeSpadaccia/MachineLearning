import numpy as np
import pandas as pd
import operator

tennis_df = pd.read_csv('tennis.csv')
tennis_df

# Tree data structure
class node():
    
    def __init__(self, 
                 name,
                 father=None,
                 attribute=None,
                 attribute_value=None,
                 root=False,
                 data=None,
                 selected_data=None,
                 processed=False, 
                 probs=[]):
        
        # variables with usefull value
        self.name = name # node name
        self.father = father # parent node
        self.root = root # defines if it is the root node
        self.data = data # data that will be passed to child node
        self.selected_data = selected_data # data used on actual node
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.childs = []
        self.processed = processed
        self.probs = probs # probability of each target
        
        # Add the node as child, if recieve a father node
        if father is not None:
            father.add_child(self)
        
    # add child node    
    def add_child(self, childs):
        self.childs.append(childs)
        
    def add_values(self, value):
        self.values.append(value)
    
# function that walks throught the tree and print it name and probs
def print_tree_nodes(node):
    if len(node.childs) == 0:
        print(f"{node.name, node.probs}")
        return
    for child in node.childs:
        print_tree_nodes(child)
    print(f"{node.name, node.probs}")
    

class decision_tree():
    
    def __init__(self):
        self.queue = []
    
    # calculate the entropy of a specific class
    # best result: 0
    # worst result: 1
    def entropy(self, values_of_a_class):
        total = np.sum(values_of_a_class)
        entropy = 0

        for i in values_of_a_class:
            p = i/total

            if p == 0:
                entropy = 0
            else:
                entropy += -p*np.log2(p)

        return entropy

    # return a list of attributes present on a dataset
    def get_attributes(self, df):
        attributes = df.columns[:-1]
        target = df.columns[-1:][0]

        labels = df[target].unique()

        print(f'Attributes: {attributes.values}')
        print(f'Target: {target}')
        print(f'Labels: {labels}')

        attributes = attributes.tolist()

        return attributes, target, labels
    
     
    # returns the best attribute to split based on entropy and gain
    def get_better_attribute(self, df, attributes, target, labels):
        total_of_lines = len(df)
        final_results = {}
        all_entropy_values = {}

        rest_attributes = len(attributes)

        final_results = {}
        all_entropy_values = {}

        for att_analisys in attributes:
            attribute = df[[att_analisys, target]]
            unique_values = attribute[att_analisys].unique()

            all_entropy_values[att_analisys] = {}

            # Calculate the entropy of the all attribute
            values_tot_att = []
            for i in labels:
                values_tot_att.append(len(df[df[target] == i]))

            att_total_entropy = self.entropy(values_tot_att)

            # Calculate the entropy for each value on attribute
            att_entropy = []
            # Entropy * (Quantity_of_attribute/Total_on_attribute)
            att_entropy_by_quantity = []

            for att_values in unique_values:
                attribute_esp = attribute[attribute[att_analisys] == att_values]

                values = []
                for i in labels:
                    values.append(len(attribute_esp[attribute_esp[target] == i]))

                # entropy of specific value of specific attribute
                entropy_value = self.entropy(values)

                att_entropy.append(entropy_value)
                all_entropy_values[att_analisys][att_values] = entropy_value

                # value to be used on gain calculation
                entropy_by_quantity = (len(attribute_esp)/total_of_lines)*entropy_value
                att_entropy_by_quantity.append(entropy_by_quantity)

            # calculate gain
            gain = 0
            for x in att_entropy_by_quantity:
                att_total_entropy = att_total_entropy - x
            gain = att_total_entropy

            final_results[att_analisys] = gain

        best_feature = max(final_results.items(), key=operator.itemgetter(1))[0]
        best_gain_values = df[best_feature].unique()

        return final_results, best_feature, all_entropy_values, best_gain_values
    
    # calculate the probability of the results provided by a node
    def get_prob(self, df, target):
        values_probs = {}
        df_target_values = df[target].value_counts()
        total_of_values = df_target_values.sum()

        for i in df_target_values.index:
            values_probs[i] = df_target_values[i]/total_of_values

        return values_probs
    
    # build the decision tree
    def build_tree(self, df, tree):
        list_of_att_to_remove = []

        attributes, target, labels = self.get_attributes(df)

        if len(attributes) < 1:
            return

        final_results, best_feature, all_entropy_values, best_gain_values = self.get_better_attribute(df, attributes, target, labels)

        if tree is None:
            tree = node(name='root', data=df, root=True)

        for feature_values in best_gain_values:
            tmp_node = node(name=str(feature_values), 
                             father=tree, 
                             attribute=best_feature,
                             attribute_value=feature_values,
                             data=df[df[best_feature] != feature_values].drop(best_feature, axis=1),
                             selected_data=df[df[best_feature] == feature_values],
                             probs=self.get_prob(df[df[best_feature] == feature_values], target))

            self.queue.append(tmp_node)
            tmp_node = None

        if len(self.queue) > 0:
            tmp_proc_node = self.queue[0]
            while len(tmp_proc_node.probs) < 2:
                del self.queue[0]
                if len(self.queue) == 0:
                    break
                tmp_proc_node = self.queue[0]

            if len(self.queue) > 0:
                del self.queue[0]

                if len(tmp_proc_node.probs) > 1:
                    self.build_tree(tmp_proc_node.selected_data, tmp_proc_node)

                tmp_proc_node = None

        return tree

    
    def fit(self, df):
        tree = None
        tree = self.build_tree(df, tree)
        
        return tree
    
    
    def get_tree_prob(self, father, row, path=None):
        if path is None:
            path = []

        if len(father.childs) == 0:
            return father.probs, path

        for node in father.childs:
            if row[node.attribute] == node.attribute_value:
                path.append(node.attribute_value)
                probs, path = self.get_tree_prob(node, row, path)
                break

        return probs, path
    
    
    def predict(self, tree, df):
        probs = []
        path = []
        for index, row in df.iterrows():
            probs_, path_ = self.get_tree_prob(tree, row)
            probs.append(probs_)
            path.append(path_)
            
        return probs, path
    
# instance of the class
dt = decision_tree()
# train of the algorithm
tree = dt.fit(tennis_df)

# print the tree
print_tree_nodes(tree)

# dataframe with a line to be predicted
df_to_predict = pd.DataFrame([['rainy','mild','high',False]],
                             columns=['outlook', 'temp', 'humidity', 'windy'])

# prediction
probs, path = dt.predict(tree, df_to_predict)
# print the result of the prediction
print(probs)