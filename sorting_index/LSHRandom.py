import numpy as np
from lshashing import LSHRandom

sample_data = np.random.randint(size=(20, 20), low=0, high=100)
point = np.random.randint(size=(1, 20), low=0, high=100)

lshashing = LSHRandom(sample_data, hash_len=4, num_tables=2)
print(lshashing.tables[0].hash_table)
print(lshashing.knn_search(sample_data, point[0], k=4, buckets=3, radius=2))

lsh_random_parallel = LSHRandom(sample_data, hash_len=4, num_tables=2, parallel=True)
print(lsh_random_parallel.knn_search(sample_data, point[0], k=4, buckets=3, radius=2, parallel=True))
