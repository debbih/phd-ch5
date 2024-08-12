"""Graph2Vec module."""

import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np ###
from scipy.stats import powerlaw, expon, halfnorm ###
import pickle ###

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        ###############################
        data_ = self.graph.edges.data("weight")
        data_df = pd.DataFrame(data_, columns=['node1', 'node2', 'weight'])
        weights = list(data_df.weight)
        min_ = min(weights)
        max_ = max(weights)
        n_nodes = len(self.nodes)
        distrib_ = halfnorm.pdf(weights, loc=min_, scale=max_/n_nodes)
        distrib_ = distrib_ / max(distrib_)
        data_df.loc[:, 'prob'] = distrib_
        #print(n_nodes, weights, distrib_)
        ###############################
        for node in self.nodes:
            ### nebs = self.graph.neighbors(node) ###
            ###############################
            
            data_df1 = data_df[data_df.node1 == node]
            data_df1.loc[:, 'nebs'] = list(data_df1.node2)
            data_df2 = data_df[data_df.node2 == node]
            data_df2.loc[:, 'nebs'] = list(data_df2.node1)
            data_df_sub = pd.concat([data_df1, data_df2])
            
            """
            min_ = min(data_df.weight)
            max_ = max(data_df.weight)
            if min_==max_:
                delta = 1
            else:
                delta = max_ - min_
            data_df.loc[:, 'prob'] = 1 - (np.array(data_df.weight) - min_) / delta
            """
            random_ = np.random.uniform(size=len(data_df_sub))
            selected = data_df_sub[data_df_sub.prob>=random_]
            nebs = list(selected.nebs) ###
            print('n nodes', len(self.nodes), 'node', node, 'n neigh', len(nebs))
            #print('n nodes', len(self.nodes), 'node', node, 'chosen around', nebs)
            ###############################
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)
    data = json.load(open(path))
    #print('data', len(data["edges"]))
    ### graph = nx.from_edgelist(data["edges"]) ###
    graph = nx.Graph()
    #graph.add_nodes_from(list(data["features"].keys()))
    graph.add_weighted_edges_from(data["edges"]) ###

    if "features" in data.keys():
        features = data["features"]
        features = {int(k): v for k, v in features.items()}
    else:
        print('error here hey') ###
        features = nx.degree(graph)
        features = {int(k): v for k, v in features}
        
    #print('----')
    #print(graph.nodes)
    #print('----')
    #print('----')
       
    return graph, features, name

def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = path2name(f)
        out.append([identifier] + list(model.dv["g_"+identifier])) ###
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path+"dim"+str(dimensions)+".csv", index=None)

def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    path = os.path.join(args.input_path, "*.json")
    graphs = glob.glob(path)
    #print(len(graphs))
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")
    
    
    #with open("document-collectionsB2.pickle", "wb") as outfile:
        #pickle.dump(document_collections, outfile)
    
    #with open("document-collectionsA.pickle", "rb") as infile:
        #document_collections = pickle.load(infile)
    #print(len(document_collections))
    
    model = Doc2Vec(vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate)
    model.build_vocab(document_collections)
    model.train(document_collections, total_examples=model.corpus_count, epochs=model.epochs)
    
    save_embedding(args.output_path, model, graphs, args.dimensions) ###

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
