# nearest_neighbor_search
Basic nearest neighbor search with [faiss](https://github.com/facebookresearch/faiss)

This setting finishes top-20 nearest neighbor search for 2 million 128-dimensional vectors in about 20 minutes, with average recall around 93%. Recall is calculated by randomly sample 10,000 vectors, do brute-force search (also with faiss) and compare the results.

Input file should be of the following format
```
[id1] [dim1] [dim2] ...
[id2] [dim1] [dim2] ...
...
```
The script right now only supports '\t' as delimiter and a fixed dimensionality of 300.
