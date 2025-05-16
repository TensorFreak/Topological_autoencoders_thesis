# Topological_autoencoders_thesis
Experiments related to my Master's thesis

Here I implemented following modifications to oriiginal TopoAE model proposed by M.Moor in [https://arxiv.org/abs/1906.00722]:
- weighting to Signature Loss for different dimentions
- minimal persistence to count only long life features
- Signature Loss with arbitrary pairwise distance metric
  
  VietorisRipsComplex can take as input precalculated distances for given point cloud,
  but in current SignatureLoss code it always calculates distances one more time since it treats first argument as pont cloud itself
   
