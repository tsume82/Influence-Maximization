import time
import networkx as nx
from node2vec import Node2Vec

from load import read_graph



# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'

# Create a graph
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)
# G = read_graph("../graphs/amazon0302.txt", directed=True)
# graph = read_graph("../graphs/wiki-Vote.txt", directed=True)
graph = read_graph("../graphs/soc-Epinions1.txt", directed=True)
start = time.time()
# Precompute probabilities and generate walks
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

# Embed
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

exec_time = time.time() - start
print("exec time {}".format(exec_time))
# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)

with open("./log", "w") as f:
	f.write("Running time: {}".format(exec_time))

