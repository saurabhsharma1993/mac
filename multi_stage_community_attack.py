import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset, Synth_Dataset, hyperparams
from models import *
from utils import *
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pickle
import math
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='polblogs', choices=['polblogs','SBM','cora_2comm', 'citeseer_2comm', 'CoauthorCS'], help='dataset')
parser.add_argument('--model', type=str, default='sage', choices=['gcn', 'sage'], help='dataset')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--switch_k', type=int, default=2, help='Number of attr perts to make at converted target.')
parser.add_argument('--perc_atkrs', type=int, default=100, help='Percentage of maxm attackers.')
parser.add_argument('--influence_thresh', type=float, default=2.7, help='Influence cutoff to make attr perts at converted target.')
parser.add_argument('--log_dir', type=str, default='', help='Weight of corruption objective.')
parser.add_argument('--exp', type=str, default='default', help='name of the experiment.')
parser.add_argument('--no_adj_pert', default=False, action='store_true')
parser.add_argument('--no_feats_pert', default=False, action='store_true')
parser.add_argument('--no_infls_pert', default=False, action='store_true')
parser.add_argument('--fixed_ip', default=False, action='store_true')
parser.add_argument('--demand', type=int, default=150, help='Number to convert')

args = parser.parse_args()
args.pert_adj, args.pert_feats = not args.no_adj_pert, not args.no_feats_pert
args.infls_pert = not args.no_infls_pert or args.fixed_ip
args.cuda = torch.cuda.is_available()
    
root_dir = './logs'
log_dir = os.path.join(root_dir,'{}'.format(args.dataset))
log_file = os.path.join(log_dir, "{}.txt".format(args.exp))

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
hyperparams = hyperparams[args.dataset][args.model]
args.influence_thresh, args.switch_k = hyperparams['influence_thresh'], hyperparams['switch_k']

for arg in vars(args):
    print_write('{}: {}'.format(arg, getattr(args, arg)),log_file)
    
## Load data and train a GCN model
if args.dataset in ['SBM','cora_2comm','citeseer_2comm','CoauthorCS']:
    data = Synth_Dataset(name=args.dataset)
else:
    data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

num_nodes = adj.shape[0]
num_feats = features.shape[1]
print_write("Num of nodes: {}, num of feats: {}".format(num_nodes,num_feats),log_file)

if args.dataset not in ['SBM','cora_2comm','citeseer_2comm','CoauthorCS']:
    features, adj, labels = torch.from_numpy(features.todense()), torch.from_numpy(adj.todense()), torch.from_numpy(labels).long()
else:
    features, adj, labels = torch.from_numpy(features).float(), torch.from_numpy(adj).float(), torch.from_numpy(labels).long()
features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
degrees = torch.sum(adj,dim=0)

idx_unlabeled = np.union1d(idx_val, idx_test)

if args.model == 'gcn':
    model = Trainable_GCN(features.shape[1],16,labels.max().item()+1)
else:
    model = Trainable_GCN(features.shape[1],16,labels.max().item()+1,use_sage=True)
model = model.to(device)
model.fit(features, adj, labels, idx_train, idx_val)

out = model.predict(features,adj)
pred_labels = out.argmax(dim=1)
test_acc = (pred_labels[idx_test] == labels[idx_test]).sum() / len(idx_test)
print_write("Test acc: {:.4f}".format(test_acc.item()),log_file)

def select_nodes(target_gcn=None,community='attack',num=10):
    
    with torch.no_grad():
        output = target_gcn.predict(features,adj)
        if community == 'attack':
            margin_dict = {}
            for idx in idx_train:
                margin = classification_margin(output[idx], labels[idx])
                if margin < 0 or labels[idx] == 1: 
                    continue
                margin_dict[idx] = margin
            sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
            attack_nodes = [x for x, y in sorted_margins]
            return attack_nodes, sorted_margins
        else:
            margin_dict = {}
            for idx in idx_test:
                margin = classification_margin(output[idx], labels[idx])
                pred = output[idx].argmax()
                if pred == 0:     # keep all nodes "classified" as targets by the ML model
                    continue 
                margin_dict[idx] = margin
            sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
            target_nodes = [x for x,y in sorted_margins]
            return target_nodes, sorted_margins

## Select attack nodes in Attack Community
attack_nodes, attack_node_margins = select_nodes(model,'attack')
if args.perc_atkrs != 100:
    tot_num = len(attack_nodes)
    sel_num = (args.perc_atkrs*tot_num)//100
    attack_nodes, attack_node_margins = attack_nodes[:sel_num], attack_node_margins[:sel_num]
    print_write('Size of attacking set is now: {}'.format(len(attack_nodes)),log_file)

## Select target nodes in Target Community
target_nodes, target_node_margins = select_nodes(model,'target')

def single_test(adj, features, target_node, source_nodes, gcn=None):
    with torch.no_grad():
        output = gcn.predict(features, adj)
        probs = torch.softmax(output[target_node],dim=0)
        margin = classification_margin(output[target_node],labels[target_node])
        preds = output.argmax(1)
        pred = preds[target_node]
        backflips = (preds[source_nodes] == 1).sum()
        acc_test = (pred == pred_labels[target_node]) if (pred_labels[target_node] == 1)\
                   else (pred != pred_labels[target_node]) 
        return acc_test.item(), backflips, pred, margin

def pick_perts(pert_inds_comb,n_perturbations,num_nodes,num_feats,pert_adj=True,pert_feats=True):
    if pert_adj and pert_feats:
        comb_grad_argmax = pert_inds_comb[:,-n_perturbations:]
        grad_argmax_adj, grad_argmax_feats = comb_grad_argmax[:,comb_grad_argmax[1,:] < num_nodes], \
                                             comb_grad_argmax[:,comb_grad_argmax[1,:] >= num_nodes]
        grad_argmax_feats[1,:] = grad_argmax_feats[1,:] - num_nodes 
    elif pert_adj and not pert_feats:
        pert_inds_adj = pert_inds_comb[:,pert_inds_comb[1,:] < num_nodes]
        grad_argmax_adj, grad_argmax_feats = pert_inds_adj[:,-n_perturbations:], None
    elif not pert_adj and pert_feats:
        pert_inds_feats = pert_inds_comb[:,pert_inds_comb[1,:] >= num_nodes]
        grad_argmax_adj, grad_argmax_feats = None, pert_inds_feats[:,-n_perturbations:]
        grad_argmax_feats[1,:] = grad_argmax_feats[1,:] - num_nodes 
    else:
        raise Exception('Must pert at least one of edges and feats.')
    return grad_argmax_adj, grad_argmax_feats

def attack_single_target(target_node,source_nodes,budget=1,pert_inds_comb=None,pert_adj=True,pert_feats=True):
    n_perturbations = int(budget)                 
    assert n_perturbations > 0, 'Number of perturbations is 0!'

    num_nodes = adj.shape[0]
    num_feats = features.shape[1]
    
    # to gather gradients on adj
    modified_adj = adj.detach().clone()
    modified_adj.requires_grad = True
    
    # to gather gradients on features
    modified_features = features.detach().clone()
    modified_features.requires_grad = True
    
    # reuse pert_inds from previous iter if available
    if pert_inds_comb is None:
        
        modified_adj.requires_grad = True
        out = model.predict(modified_features,modified_adj)
        loss = F.cross_entropy(out[[target_node]], pred_labels[[target_node]])    
        
        grad_adj, grad_feats = torch.autograd.grad(loss, [modified_adj, modified_features])
        
        with torch.no_grad():
            grad_adj = (grad_adj + grad_adj.T) * (-modified_adj + 1)            # deletions get zeroed out
            grad_adj[target_node,target_node] = -10                                               # self loops disallowed
            
            fudge = 2*torch.max(torch.max(torch.abs(grad_adj)),torch.max(torch.abs(grad_feats)))
            grad_adj[source_nodes,:] = grad_adj[source_nodes,:] + fudge    # edge perts from source nodes have highest gradient
            grad_feats[source_nodes,:] = grad_feats[source_nodes,:] + fudge    # attribute perts from source nodes have highest gradient

            comb_grad = torch.cat((grad_adj,grad_feats),dim=1)
            grad_sort_comb, pert_inds_comb = torch.sort(comb_grad.flatten())         
            pert_inds_comb = torch.stack([pert_inds_comb//(num_nodes+num_feats), pert_inds_comb%(num_nodes+num_feats)]).long()
            
            # filter out non-positive gradients, such as deletions and useless insertions
            nnp_inds_comb = grad_sort_comb <= 0
            pert_inds_comb = pert_inds_comb[:,~nnp_inds_comb]
            
            # only check lower triangular indices of adj
            select = torch.logical_or(pert_inds_comb[0,:] < pert_inds_comb[1,:], pert_inds_comb[1,:] >= num_nodes)       
            pert_inds_comb = pert_inds_comb[:,select]
        
    grad_argmax, grad_feats_argmax = pick_perts(pert_inds_comb,n_perturbations,num_nodes,num_feats,pert_adj=pert_adj,pert_feats=pert_feats)
            
    # make updates
    if pert_adj:
        value = -modified_adj[grad_argmax[0],grad_argmax[1]] + 1       
        modified_adj.data[grad_argmax[0],grad_argmax[1]] += value
        modified_adj.data[grad_argmax[1],grad_argmax[0]] += value
    
    if pert_feats:
        value = -2*modified_features[grad_feats_argmax[0],grad_feats_argmax[1]] + 1       
        modified_features.data[grad_feats_argmax[0],grad_feats_argmax[1]] += value
    
    acc, backflips, pred, margin = single_test(modified_adj, modified_features, target_node, source_nodes, gcn=model)
    
    return acc, pred, margin, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips
    
## bisection method to compute budget    
def bin_search_fga(target_node, source_nodes, pert_inds_comb = None, pert_adj = True, pert_feats = True):
    l_budget, u_budget = 0, degrees[target_node].item()
    l_acc, backflips, _, _ = single_test(adj, features, target_node, source_nodes, gcn=model)
    if l_acc == 0:
        modified_adj, modified_features = adj.detach().clone(), features.detach().clone()
        modified_adj.requires_grad, modified_features.requires_grad = True, True        
        return l_budget, None, None, modified_adj, modified_features, backflips
    u_acc = 1
    max_budget = torch.max(degrees)   
    # find upper bound on required budget.
    while u_acc == 1:
        if u_budget > max_budget.item():
            return math.inf, None, None, None, None, None    
        u_acc, _, _, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips = attack_single_target(target_node,source_nodes,u_budget,pert_inds_comb,pert_adj,pert_feats)
        if u_acc == 1:
            u_budget = 2*u_budget
    while u_budget - l_budget > 1:    
        c_budget = (l_budget + u_budget) // 2
        c_acc, _, _, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips = attack_single_target(target_node,source_nodes,c_budget,pert_inds_comb,pert_adj,pert_feats)
        if c_acc == 1:
            l_budget = c_budget
        else:
            u_budget = c_budget
    u_acc, _, _, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips = attack_single_target(target_node,source_nodes,u_budget,pert_inds_comb,pert_adj,pert_feats)
    del(pert_inds_comb)
    return u_budget, grad_argmax, grad_feats_argmax, modified_adj, modified_features, backflips         

## Compute influence for a target in Community 2
def target_influence_lookahead(target_node, target_set, modified_adj, modified_features, switch_k = 2):
    
    out = model.predict(modified_features,modified_adj)  
    obj = torch.sum(out[target_set,0] - out[target_set,1]) 
    # get gradient on target attributes
    grad_feats = torch.autograd.grad(obj, modified_features)[0][target_node,:]                                          
    grad_feats = grad_feats * (-2*modified_features[target_node,:] + 1)
    grad_feats_sort, sort_inds = torch.sort(grad_feats)
    
    # filter nnp gradient
    nnp_inds = grad_feats_sort <= 0
    grad_feats_sort, sort_inds = grad_feats_sort[~nnp_inds], sort_inds[~nnp_inds]  
    
    # make top-k attr perts of target node
    grad_feats_argmax = sort_inds[-switch_k:]
    value = -2 * modified_features[target_node, grad_feats_argmax] + 1       
    pert_features = modified_features.detach().clone()
    pert_features[target_node, grad_feats_argmax] += value
    
    # now gradient of 2nd order perts on edges
    out_lookahead = model.predict(pert_features,modified_adj)
    obj = torch.sum(out_lookahead[target_set,0] - out_lookahead[target_set,1]) 
    
    grad_adj = torch.autograd.grad(obj, modified_adj)[0]
    grad_adj[target_node,target_node] = -10                          # no self loops
    grad_adj = (grad_adj + grad_adj.T)[target_node,:]
    grad_adj = grad_adj * (-modified_adj[target_node,:] + 1)         # considering insertions only
    grad_adj = grad_adj[target_set]
    
    # filter nnp gradient
    nnp_inds = grad_adj <= 0
    grad_adj = grad_adj[~nnp_inds]
    
    # sum all 2nd order nnn gradients
    influence = torch.sum(grad_adj).item() / len(target_set)
        
    return influence, grad_feats_argmax, pert_features

## find min budget target with highest influence
def score(target_nodes, source_nodes, switch_k = 2):
    
    best_budget, best_influence, best_target_perts = 1e13, 0, None
    best_target_node, best_grad_argmax, best_grad_feats_argmax = None, None, None
    best_modified_adj, best_modified_features, best_pert_features = None, None, None
    best_backflips = 1e13

    for ind, target_node in enumerate(target_nodes):

        budget, grad_argmax, grad_feats_argmax, modified_adj, modified_features, backflips = bin_search_fga(target_node,source_nodes,None,args.pert_adj,args.pert_feats)

        if budget == math.inf or budget > best_budget:
            continue
        
        target_set = deepcopy(target_nodes)
        target_set.remove(target_node) 
        
        if args.infls_pert: 
            influence, target_perts, pert_features = target_influence_lookahead(target_node, target_set, modified_adj, modified_features, switch_k)
        else:
            influence, target_perts, pert_features = 0, None, None

        if budget < best_budget or (budget == best_budget and influence > best_influence and backflips <= best_backflips):
        
            best_budget = budget
            best_target_node, best_grad_argmax, best_grad_feats_argmax = target_node, grad_argmax, grad_feats_argmax
            best_influence, best_target_perts = influence, target_perts
            best_modified_adj = modified_adj
            # make attribute perts at converted target node if its influence is good. or if using fixed ip
            best_modified_features = pert_features if args.infls_pert and ((influence != 0 and influence >= args.influence_thresh) or args.fixed_ip) else modified_features
            best_backflips = backflips
    return best_target_node, best_budget, best_grad_argmax, best_grad_feats_argmax, best_influence, best_target_perts, best_modified_adj, best_modified_features

## convert selected target and make perturbations
def local_attack_step(target_nodes,source_nodes,switch_k=2,pert_adj=True,pert_feats=True):
    
    # score and pick node to flip and corrupt
    target_node, budget, grad_argmax, grad_feats_argmax, influence, target_perts, modified_adj, modified_features = score(target_nodes, source_nodes, switch_k)
    
    # check if any target node is flippable at all
    if budget == math.inf or budget == 1e13:
        return target_nodes, source_nodes, None, None, None, None, None, None
    elif budget > 0:
        # make updates
        global adj 
        adj = modified_adj.detach().clone()
        global features
        features = modified_features.detach().clone()
    else: 
        pass

    all_ips = {}
    # print attribute perts at converted target node if its influence is good
    if args.infls_pert and ((influence != 0 and influence >= args.influence_thresh) or args.fixed_ip):
        print_write('Influence is good at node: {}'.format(target_node),log_file)
        print_write('Influential perturbations at node {}.'.format(target_node),log_file)
        print_write(target_perts.data.cpu().numpy(),log_file)
        budget = budget + switch_k # add perts at converted target to budget
        all_ips[target_node] = target_perts
    
    print_write("Target node: {}, Budget: {}".format(target_node,budget),log_file)
    print_write("Edge perturbations.",log_file)
    print_write(grad_argmax.data.cpu().numpy(),log_file)
    print_write("Feature perturbations.",log_file)
    print_write(grad_feats_argmax.data.cpu().numpy(),log_file)

    # recompute set of attacking and target nodes.
    new_target_nodes, _ = select_nodes(model,'target')
    converted_nodes = list(set.difference(set(target_nodes),set(new_target_nodes)))

    print_write("Converted nodes.",log_file)
    print_write(converted_nodes,log_file)
 
    converted_nodes.remove(target_node)
    converted_nodes = [target_node] + converted_nodes # for tracking purpose, keep target at index 0.
    target_nodes = new_target_nodes
    source_nodes = source_nodes + converted_nodes
    
    backflips = list(set.intersection(set(target_nodes),set(source_nodes)))
    for node in backflips:
        print_write('Node {} backflipped.'.format(node),log_file)
        source_nodes.remove(node)

    return target_nodes, source_nodes, converted_nodes, budget, influence, all_ips, grad_argmax, grad_feats_argmax

def to_numpy(rand_cuda_var):
    if rand_cuda_var == None:
        return rand_cuda_var
    else:
        return rand_cuda_var.data.cpu().numpy()

## output sequence of attack edges, budgets, attribute switches, and converted target nodes.
if __name__ == '__main__':
    num_nodes = adj.shape[0]
    converted_nodes, budgets, influences, attack_edges, attribute_switches, infl_perts, num_converted = [], [], [], [], [], [], []
    other_target_nodes = (labels[idx_train] == 1).sum() + (labels[idx_val] == 1).sum()
    start_time = time()
    init_target_size = len(target_nodes)
    last_best, time_since = init_target_size, 0
    while True:
        sys.stdout.flush()
        target_nodes, attack_nodes, converted_node_list, budget, influence, all_ips, grad_argmax, grad_feats_argmax = local_attack_step(target_nodes, attack_nodes, args.switch_k, args.pert_adj, args.pert_feats)
        if converted_node_list == None:
            print_write('Attack failed. No target node is flippable.',log_file)
            break
        converted_nodes.append(converted_node_list) 
        budgets.append(budget)                                                                 
        influences.append(influence)
        attack_edges.append(to_numpy(grad_argmax))
        attribute_switches.append(to_numpy(grad_feats_argmax))
        infl_perts.extend([(k,to_numpy(v)) for (k,v) in all_ips.items()])
        num_converted.append(init_target_size - len(target_nodes))
        
        # if target set size doesn't improve for 40 steps, stop.
        if len(target_nodes) < last_best:
            last_best, time_since = len(target_nodes), 0
        else:
            time_since += 1
        if time_since >= 40:
            print_write('Failure. Attack didn\'t decrease target size for 40 steps.',log_file)
            break

        end_time = time()
        runtime = end_time - start_time
        print_write("Time taken for {} steps: {:.2f} secs. Estimated total: {:.2f} secs.".format(len(budgets),runtime,runtime*(args.demand)/len(budgets)),log_file)
        
        if init_target_size - len(target_nodes) >= args.demand: # stopping criterion
            break

    end_time = time()
    runtime = end_time - start_time
    total_budget = sum(budgets)
    
    print_write("Time taken to attack: {:.2f} seconds".format(runtime),log_file)
    print_write("Total budget: {}".format(total_budget),log_file)
    
    results = {'args': args,\
                'converted_nodes': converted_nodes,\
    			'budgets': budgets,\
                'total_budget': total_budget,\
    			'influences': influences,\
    			'attack_edges': attack_edges,\
                'runtime': runtime,\
    			'attribute_switches': attribute_switches,\
                'infl_perts': infl_perts,\
                'num_converted': num_converted}

    with open(os.path.join(log_dir,'{}.pkl'.format(args.exp)), 'wb') as f:
    	pickle.dump(results,f)   