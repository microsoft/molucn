# Code adapted from tensorflow to pytorch from https://github.com/google-research/graph-attribution/tree/main/graph_attribution
import numpy as np
from xaikenza.feat_attribution.explainer_base import Explainer
from torch_geometric.data import Data
from copy import deepcopy
import numpy as np
import scipy.special
import torch

def normalize(a, min, max):
    return (a - min)/ (max - min)
    
    
class SHAP(Explainer):
    """ KernelSHAP explainer - adapted to GNNs
    Explains only node features
    """
    def __init__(self, device, model, num_features):
        super(SHAP, self).__init__(device, model)
        # number of nonzero features - for each node index
        self.M = num_features
        self.neighbours = None
        self.F = self.M

    def explain_graph(self, graph, model=None, num_samples=10):
        """
        :param node_index: index of the node of interest
        :param hops: number k of k-hop neighbours to consider in the subgraph around node_index
        :param num_samples: number of samples we want to form GraphSVX's new dataset 
        :return: shapley values for features that influence node v's pred
        """
        
        if model == None:
            model = self.model
        # Compute true prediction of model, for original instance
        with torch.no_grad():
            self.model = self.model.to(self.device)
        tmp_graph = graph.clone().to(self.device)
        # Determine z => features whose importance is investigated
        # Decrease number of samples because nodes are not considered
        num_samples = num_samples//3
        
        node_weights = np.zeros(tmp_graph.x.size(0))
        # Consider all features (+ use expectation like below)
        # feat_idx = torch.unsqueeze(torch.arange(self.F), 1)

        for node_index in range(tmp_graph.num_nodes):
            # Sample z - binary vector of dimension (num_samples, M)
            z_ = torch.empty(num_samples, self.M).random_(2)
            # Compute |z| for each sample z
            s = (z_ != 0).sum(dim=1)

            # Define weights associated with each sample using shapley kernel formula
            weights = self.shapley_kernel(s)

            # Create dataset (z, f(z')), stored as (z_, fz)
            # Retrive z' from z and x_v, then compute f(z')
            fz = self.compute_pred(tmp_graph, node_index, num_samples, z_)

            # OLS estimator for weighted linear regression
            phi, base_value = self.OLS(z_, weights, fz)  # dim (M*num_classes)
            
            node_weights[node_index] = phi.sum()
        return node_weights

    def shapley_kernel(self, s):
        """
        :param s: dimension of z' (number of features + neighbours included)
        :return: [scalar] value of shapley value 
        """
        shap_kernel = []
        # Loop around elements of s in order to specify a special case
        # Otherwise could have procedeed with tensor s direclty
        for i in range(s.shape[0]):
            a = s[i].item()
            # Put an emphasis on samples where all or none features are included
            if a == 0 or a == self.M:
                shap_kernel.append(1000)
            elif scipy.special.binom(self.M, a) == float('+inf'):
                shap_kernel.append(1/self.M)
            else:
                shap_kernel.append(
                    (self.M-1)/(scipy.special.binom(self.M, a)*a*(self.M-a)))
        return torch.tensor(shap_kernel)

    def compute_pred(self, graph, node_index, num_samples, z_):
        """
        Variables are exactly as defined in explainer function, where compute_pred is used
        This function aims to construct z' (from z and x_v) and then to compute f(z'), 
        meaning the prediction of the new instances with our original model. 
        In fact, it builds the dataset (z, f(z')), required to train the weighted linear model.
        :return fz: probability of belonging to each target classes, for all samples z'
        fz is of dimension N*C where N is num_samples and C num_classses. 
        """
        # This implies retrieving z from z' - wrt sampled neighbours and node features
        # We start this process here by storing new node features for v and neigbours to
        # isolate
        X_v = torch.zeros([num_samples, self.F])

        # Init label f(z') for graphshap dataset - consider all classes
        fz = torch.zeros(num_samples)

        # Do it for each sample
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Features where z_j == 1 are kept, others are set to 0
            for j in range(self.F):
                if z_[i, j].item() == 1:
                    X_v[i, j] = 1

            # Change feature vector for node of interest
            X = deepcopy(graph.x)
            X[node_index, :] = X_v[i, :]

            # Apply model on (X,A) as input.
            with torch.no_grad():
                fz[i] = self.model(graph)
            
        return fz

    def OLS(self, z_, weights, fz):
        """
        :param z_: z - binary vector  
        :param weights: shapley kernel weights for z
        :param fz: f(z') where z is a new instance - formed from z and x
        :return: estimated coefficients of our weighted linear regression - on (z, f(z'))
        phi is of dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(0.00001 * np.random.randn(tmp.shape[1])))
        phi = np.dot(tmp, np.dot(
            np.dot(z_.T, np.diag(weights)), fz.cpu().detach().numpy()))

        # Test accuracy
        # y_pred=z_.detach().numpy() @ phi
        #	print('r2: ', r2_score(fz, y_pred))
        #	print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]