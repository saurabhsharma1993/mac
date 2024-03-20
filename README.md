# Electioneering the Network: Dynamic Multi-Step Attacks for Community Canvassing 
<img src="https://github.com/saurabhsharma1993/mac/blob/main/teaser.png" width="500">
The problem of online social network manipulation for community canvassing is of real concern in today's world. Motivated by the study of voter models, opinion and polarization dynamics on networks, we model community canvassing as a dynamic process over a network enabled via gradient-based attacks on GNNs. Existing attacks on GNNs are all single-step and do not account for the dynamic cascading nature of information diffusion in networks. We consider the realistic scenario where an adversary uses a GNN as a proxy to predict and manipulate voter preferences, especially uncertain voters. Gradient-based attacks on the GNN inform the adversary of strategic manipulations that can be made to proselytize targeted voters. In particular, we explore _minimum budget attacks for community canvassing_ (MBACC). We show that the MBACC problem is NP-Hard and propose Dynamic Multi-Step Adversarial Community Canvassing (MAC) to address it. MAC makes dynamic local decisions based on the heuristic of low budget and high second-order influence to convert and perturb target voters. MAC is a dynamic multi-step attack that discovers low-budget and high-influence targets from which efficient cascading attacks can happen. We evaluate MAC against single-step baselines on the MBACC problem with multiple underlying networks and GNN models. Our experiments show the superiority of MAC which is able to discover efficient multi-hop attacks for adversarial community canvassing.

# Demo
We provide a demo of our research in the form a jupyter-notebook, to make our work more easily accessible. To use the demo, simply download ___demo.ipynb___ and run it on a Google Colab GPU instance. The demo works for the Polblogs dataset; for the other datasets please download the data as outlined below. 

# Running locally
1. Clone the conda environment
```
conda env create -f environment.yml
```
2. Download the Cora-2, Citeseer-2, CoauthorCS-2 and SBM datasets here:
```
https://drive.google.com/drive/folders/1_QQyfUzZ75zmadLU-RhAgNYzaLzBnBMb?usp=sharing
```
3. Run the MAC (Dynamic IP) attack,
```
python multi_state_community_attack.py --dataset [name_of_dataset] --exp [name_of_exp] 
```
The experiment logs and results will be dumped in,
```
./logs/[name_of_dataset]/[name_of_exp]
```
# Citing 
If you use this code, please cite our work:
```
@misc{sharma2024electioneering,
      title={Electioneering the Network: Dynamic Multi-Step Adversarial Attacks for Community Canvassing}, 
      author={Saurabh Sharma and Ambuj SIngh},
      year={2024},
      eprint={2403.12399},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
