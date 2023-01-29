import pandas as pd
import numpy as np
import warnings
import random
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


class BN(object):
    """
    Bayesian Network implementation with sampling methods as a class
    
    Attributes
    ----------
    n: int
        number of variables
        
    G: dict
        Network representation as a dictionary. 
        {variable:[[children],[parents]]} # You can represent the network in other ways. This is only a suggestion.

    topological order: list
        topological order of the nodes of the graph

    CPT: list
        CPT Table
    """

    def __init__(self, graph, CPT) -> None:
        ############################################################
        # Initialzie Bayesian Network                              #
        # (1 Points)                                               #
        ############################################################

        # Your code
        self.graph = graph
        self.CPT = CPT
        self.sorted_nodes = None
        self.parents = {key: [] for key in self.graph.keys()}
        for key, value in self.graph.items():
            for child in value:
                self.parents[child].append(key)
        joint = None
        for key, value in self.CPT.items():
            if joint is None:
                joint = value
            else:
                t2 = CPT[key]
                join_columns = set(joint.keys()).intersection(set(t2.keys())) - set('P')
                joint = pd.merge(joint, t2, on=list(join_columns), how='inner')
                joint['P'] = joint['P_x'] * joint['P_y']
                joint.drop(['P_x', 'P_y'], axis=1, inplace=True)
        self.joint = joint

    def cpt(self, node, value=None) -> dict:
        """
        This is a function that returns cpt of the given node
        
        Parameters
        ----------
        node:
            a variable in the bayes' net
            
        Returns
        -------
        result: dict
            {value1:{{parent1:p1_value1, parent2:p2_value1, ...}: prob1, ...}, value2: ...}
        """
        ############################################################
        # (3 Points)                                               #
        ############################################################   
        # Your code
        return self.CPT[node]

    def pmf(self, query, evidence) -> float:
        """
        This function gets a variable and its value as query and a list of evidences and returns probability mass function P(Q=q|E=e)
        
        Parameters
        ----------
        query:
            a variable and its value
            e.g. ('a', 1)
        evidence:
            list of variables and their values
            e.g. [('b', 0), ('c', 1)]
        
        Returns
        -------
        PMF: float
            P(query|evidence)
        """
        ############################################################
        # (3 Points)                                               #
        ############################################################
        set_vars = set()
        #Variable Elimination The hardest challenge of the code by FARRR :DDDD#
        if self.sorted_nodes is None:
            self.set_topological_order()
        result = self.cpt(self.sorted_nodes[-1])
        for node in reversed(self.sorted_nodes):
            related_CPTS = []
            for child in self.graph[node]:
                if child not in set_vars:
                    set_vars.add(child)
                    related_CPTS.append(self.cpt(child))
            if node not in set_vars:
                set_vars.add(node)
                related_CPTS.append(self.cpt(node))
            if len(related_CPTS) > 1:
                joint = None
                for cpt in related_CPTS:
                    if joint is None:
                        joint = cpt
                    else:
                        t2 = cpt
                        join_columns = set(joint.keys()).intersection(set(t2.keys())) - set('P')
                        joint = pd.merge(joint, t2, on=list(join_columns), how='inner')
                        joint['P'] = joint['P_x'] * joint['P_y']
                        joint.drop(['P_x', 'P_y'], axis=1, inplace=True)
                result = joint
            elif len(related_CPTS) == 0:
                continue
            else:
                #merge result and related_CPT
                t2 = related_CPTS[0]
                join_columns = set(result.keys()).intersection(set(t2.keys())) - set('P')
                if len(set_vars) != 1:
                    result = pd.merge(result, t2, on=list(join_columns), how='inner')
                    result['P'] = result['P_x'] * result['P_y']
                    result.drop(['P_x', 'P_y'], axis=1, inplace=True)
                if node in evidence or node in query:
                    continue
                group_by_columns = set(result.keys()) - set(node) - set('P')
                result = result.groupby(by=list(group_by_columns), as_index=False).sum()
                result = result.drop(columns=[node])
        mask = [True] * len(result)
        for e, val in evidence.items():
            mask = mask & (result[e] == val)
        result = result[mask]
        mask = [True] * len(result)
        #Normalize
        result['P'] = result['P'] / result['P'].sum()
        for e, val in query.items():
            mask = mask & (result[e] == val)
        result = result[mask]
        return result['P'].values[0]


        # joint = None
        # for key, value in self.CPT.items():
        #     if joint is None:
        #         joint = value
        #     else:
        #         t2 = CPT[key]
        #         join_columns = set(joint.keys()).intersection(set(t2.keys())) - set('P')
        #         joint = pd.merge(joint, t2, on=list(join_columns), how='inner')
        #         joint['P'] = joint['P_x'] * joint['P_y']
        #         joint.drop(['P_x', 'P_y'], axis=1, inplace=True)
        # mask = [True] * len(joint)
        # for e, val in evidence.items():
        #     mask = mask & (joint[e] == val)
        # table = joint[mask]
        # Z = table['P'].sum()
        # table.loc[:, 'P'] = table['P'] / Z
        # mask = [True] * len(table)
        # for q, val in query.items():
        #     mask = mask & (table[q] == val)
        # table = table[mask]
        # return table['P'].sum()

    def sampling(self, query, evidence, sampling_method, num_iter, num_burnin=1e2) -> float:
        """
        Parameters
        ----------
        query: list
            list of variables an their values
            e.g. [('a', 0), ('e', 1)]
        evidence: list
            list of observed variables and their values
            e.g. [('b', 0), ('c', 1)]
        sampling_method:
            "Prior", "Rejection", "Likelihood Weighting", "Gibbs"
        num_iter:
            number of the generated samples 
        num_burnin:
            (used only in gibbs sampling) number of samples that we ignore at the start for gibbs method to converge
            
        Returns
        -------
        probability: float
            approximate P(query|evidence) calculated by sampling
        """
        ############################################################
        # (27 Points)                                              #
        #     Prior sampling (6 points)                            #
        #     Rejection sampling (6 points)                        #
        #     Likelihood weighting (7 points)                      #
        #     Gibbs sampling (8 points)                      #
        ############################################################
        if sampling_method == "Prior":
            return self.prior_sample(query, evidence, num_iter)
        elif sampling_method == "Rejection":
            return self.rejection_sample(query, evidence, num_iter)
        elif sampling_method == "Likelihood Weighting":
            return self.likelihood_sample(query, evidence, num_iter)
        elif sampling_method == "Gibbs":
            return self.gibbs_sample(query, evidence, num_iter, num_burnin)

        # Your code
        pass

    def prior_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            prior samples
        """
        df_dictionary = {key: [] for key in self.graph.keys()}
        for i in tqdm(range(num_iter)):
            sample = self.get_prior_sample()
            # add all sample to df_dictionary
            for key, value in sample.items():
                df_dictionary[key].append(value)
        samples = pd.DataFrame(df_dictionary)
        mask = [True] * len(samples)

        for e, val in evidence.items():
            mask = mask & (samples[e] == val)
        valids = len(samples[mask])
        for q, val in query.items():
            mask = mask & (samples[q] == val)
        table = samples[mask]
        true = len(table)
        return true / valids

    def sample_consistent_with_evidence(self, sample, evidence):
        """
            To check if a sample is consistent with evidence or not?

            Parameters
            ----------
            sample:
                a sample
            evidence:
                evidence set
            
            Returns
            -------
            True if the sample is consistent with evidence, False otherwise.
        """
        for e, val in evidence.items():
            if sample[e] != val:
                return False
        return True

    def sample_consistent_with_query(self, sample, query):
        """
            To check a sample is consistent with query or not?

            Parameters
            ----------
            sample:
                a sample
            evidence:
                query set
            
            Returns
            -------
            True if the sample is consistent with query, False otherwise.
        """
        for q, val in query.items():
            if sample[q] != val:
                return False
        return True

    def get_prior_sample(self):
        """
            Returns
            -------
            Returns a set which is the prior sample. 
        """
        if self.sorted_nodes is None:
            self.set_topological_order()
        set_vars = {}
        # for e in evidences:
        #     set_vars[e[0]] = e[1]
        for node in self.sorted_nodes:
            nodes_cpt = self.cpt(node)
            mask = [True] * len(nodes_cpt)
            for parent in self.parents[node]:
                mask = mask & (nodes_cpt[parent] == set_vars[parent])
            nodes_cpt = nodes_cpt[mask]
            set_vars[node] = np.random.choice(nodes_cpt[node], p=nodes_cpt['P'])
        return set_vars

    def rejection_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            rejection samples
        """
        df_dictionary = {key: [] for key in self.graph.keys()}
        for i in tqdm(range(num_iter)):
            if self.sorted_nodes is None:
                self.set_topological_order()
            sample = {}
            # for e in evidences:
            #     set_vars[e[0]] = e[1]
            reject = False
            # change evidence to dict
            evidence_dict = {}
            for e, val in evidence.items():
                evidence_dict[e] = val
            for node in self.sorted_nodes:
                nodes_cpt = self.cpt(node)
                mask = [True] * len(nodes_cpt)
                for parent in self.parents[node]:
                    mask = mask & (nodes_cpt[parent] == sample[parent])
                nodes_cpt = nodes_cpt[mask]
                new_value = np.random.choice(nodes_cpt[node], p=nodes_cpt['P'])
                if node in evidence_dict and new_value != evidence_dict[node]:
                    reject = True
                    break
                sample[node] = new_value
            # add all sample to df_dictionary
            if not reject:
                for key, value in sample.items():
                    df_dictionary[key].append(value)
        samples = pd.DataFrame(df_dictionary)
        mask = [True] * len(samples)

        for e, val in evidence.items():
            mask = mask & (samples[e] == val)
        valids = len(samples[mask])
        for q, val in query.items():
            mask = mask & (samples[q] == val)
        table = samples[mask]
        true = len(table)
        return true / valids

    def likelihood_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            likelihood samples
        """
        df_dictionary = {key: [] for key in self.graph.keys()}
        total = 0
        true = 0
        if self.sorted_nodes is None:
            self.set_topological_order()
        for i in tqdm(range(num_iter)):
            sample = {key : evidence[key] for key in evidence.keys()}
            evidence_dict = {}
            for e, val in evidence.items():
                evidence_dict[e] = val
            w = 1.0
            for node in self.sorted_nodes:
                if node in evidence_dict:
                    continue
                nodes_cpt = self.cpt(node)
                mask = [True] * len(nodes_cpt)
                for parent in self.parents[node]:
                    mask = mask & (nodes_cpt[parent] == sample[parent])
                nodes_cpt = nodes_cpt[mask]
                new_value = np.random.choice(nodes_cpt[node], p=nodes_cpt['P'])
                sample[node] = new_value
            for node in evidence_dict:
                nodes_cpt = self.cpt(node)
                mask = [True] * len(nodes_cpt)
                for parent in self.parents[node]:
                    mask = mask & (nodes_cpt[parent] == sample[parent])
                nodes_cpt = nodes_cpt[mask]
                w *= nodes_cpt[nodes_cpt[node] == evidence_dict[node]]['P'].values[0]
            total = total + w
            if self.sample_consistent_with_query(sample, query):
                if self.sample_consistent_with_evidence(sample, evidence):
                    true = true + w
        return true / total

    def gibbs_sample(self, query, evidence, num_iter, num_burnin):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            gibbs samples
        """
        num_burnin = int(num_burnin)
        current = {key: 1 if np.random.rand(1) < 0.5 else 1 for key in self.graph.keys()}
        for e, val in evidence.items():
            current[e] = val
        perm = set(current.keys()) - set(evidence.keys())
        true_values = 0
        total = 0
        for i in tqdm(range(num_iter + int(num_burnin))):
            for node in perm:
                temp = self.joint
                for s, val in current.items():
                    if s != node:
                        temp = temp[temp[s] == val]
                prob_one = temp[temp[node] == 1]['P'].sum() / temp['P'].sum()
                if np.random.rand(1) < prob_one:
                    current[node] = 1
                else:
                    current[node] = 0
            if i > num_burnin:
                total += 1
                if self.sample_consistent_with_query(current, query) and self.sample_consistent_with_evidence(current, evidence):
                    true_values += 1
        print(true_values, total)
        return true_values / total

    def topological_sort(self, node, visited):
        """
            This function wants to make a topological sort of the graph and set the topological_order parameter of the class.

            Parameters
            ----------
            node:
                the list of nodes
            visited:
                the list of visited(1)/not visited(0) nodes

        """
        visited[node] = True
        for i in self.graph[node]:
            if not visited[i]:
                self.topological_sort(i, visited)
        self.sorted_nodes.append(node)

    def set_topological_order(self):
        """
            This function calls topological sort function and set the topological sort.
        """
        if self.sorted_nodes is not None:
            return
        if self.sorted_nodes is None:
            self.sorted_nodes = []
        visited = {node: False for node in self.graph.keys()}
        for i in self.graph.keys():
            if not visited[i]:
                self.topological_sort(i, visited)
        self.sorted_nodes = self.sorted_nodes[::-1]

    def all_parents_visited(self, node, visited) -> bool:
        """
            This function checks if all parents are visited or not?

            Parameters
            ----------
            node:
                the list of nodes
            visited:
                the list of visited(1)/not visited(0) nodes

            Return
            ----------
            return True if all parents of node are visited, False otherwise.
        """
        pass

    def remove_nonmatching_evidences(self, evidence, factors):
        pass

    def join_and_eliminate(self, var, factors, evidence):
        pass

    def get_joined_factor(self, var_factors, var, evidence):
        pass

    def get_rows_factor(self, factor, var, evidence, values, variables_in_joined_factor):
        pass

    def get_var_factors(self, var, factors):
        pass

    def get_variables_in_joined_factor(self, var_factors, var, evidence):
        pass

    def get_join_all_factors(self, factors, query, evidence):
        pass

    def get_row_factor(self, factor, query_vars, evidence, values):
        pass

    def normalize(self, joint_factor):
        pass


graph = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": ["F"], "E": ["C"], "F": []}

CPT = {
    "A": pd.DataFrame({"A": [1, 0], "P": [0.8, 0.2]}),
    "B": pd.DataFrame({"B": [1, 1, 0, 0], "A": [1, 0, 1, 0], "P": [0.3, 0.9, 0.7, 0.1]}),
    "D": pd.DataFrame({"D": [1, 1, 0, 0], "B": [1, 0, 1, 0], "P": [0.2, 0.25, 0.8, 0.75]}),
    "E": pd.DataFrame({"E": [1, 1, 0, 0], "B": [1, 0, 1, 0], "P": [0.15, 0.3, 0.85, 0.7]}),
    "C": pd.DataFrame({"C": [1, 1, 1, 1, 0, 0, 0, 0], "A": [1, 1, 0, 0, 1, 1, 0, 0], "E": [1, 0, 1, 0, 1, 0, 1, 0],
                       "P": [0.1, 0.3, 0.05, 0.6, 0.9, 0.7, 0.95, 0.4]}),
    "F": pd.DataFrame({"F": [1, 1, 1, 1, 0, 0, 0, 0], "C": [1, 1, 0, 0, 1, 1, 0, 0], "D": [1, 0, 1, 0, 1, 0, 1, 0],
                       "P": [0.05, 0.5, 0.15, 0.7, 0.95, 0.5, 0.85, 0.3]}),

}

#My tests
# bn = BN(graph, CPT)
# bn.set_topological_order()
# query = {"F": 1}
# evidence = {"B": 1, "C": 0}
# print(bn.pmf(query, evidence))
#bn = BN(graph, CPT)
# query = {'F': 0, 'E':1}
# evidence = {'A': 1, 'D': 0}
# print(bn.pmf(query, evidence))
# print(bn.rejection_sample(query, evidence, 1000))
# print(bn.prior_sample(query, evidence, 1000))
# print(bn.gibbs_sample(query, evidence, 1000, 100))
# print(bn.likelihood_sample(query, evidence, 2000))
#print(bn.sorted_nodes)
########### Testing PMF
# pd.options.mode.chained_assignment = None
# bn = BN(graph, CPT)
# bn.set_topological_order()
# prob = bn.pmf({"A": 1, "B":1}, {"C": 0})
# print(prob)
# prob = bn.pmf({"B": 1}, {"A": 1})
# print(prob)

# ########### Testing prior_sample
# import time
# start_time = time.time()
# print(bn.sampling([("A", 1)], [("B", 1), ("C", 0)], "Prior", 1000))
# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()
# print(bn.sampling([("A", 1)], [("B", 1), ("C", 0)], "Rejection", 1000))
# print("--- %s seconds ---" % (time.time() - start_time))
# print(bn.sorted_nodes)
