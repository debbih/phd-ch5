*Chapter 5 - "Blockchain Transactions and Species of Traders" from Deborah Miori's PhD Thesis*

- After downloading updated data from Uniswap Subgraph, as highlighted in the above manuscript (Chapter 5 of the thesis), one can proceed to subset data to liquidity pools with some minimum levels of activity.
  This can be achieved by following the constraints as imposed in Notebook "[0] subset relevant data.ipynb".
- Then, the interconnectedness among pools is computed to define a more focused final set of pools with maximised interactions.
  This is mainly done within Notebook "[1] subset graph enhancement.ipynb".
- However, the computation of "bridges" from related actions within same Uniswap transactions is done within "[1a] bridges-computation.py".
- Data on liquidity takers activity during Jan-Jun 2022 is extracted for the above focused set of pools and provided in file "final_poolsA.pkl".
- Finally, our modified graph2vec algorithm for the clustering of liquidity takers relies on the following files:
  - "[2] create trader graph.py" to build the characteristic graph of transaction for each liquidity taker
  - "[3] param_parser.py" and "[3a] graph2vec.py", which are our newly proposed versions of the graph2vec algorithm  (see https://github.com/annamalai-nr/graph2vec_tf.git) to apply to our case.
