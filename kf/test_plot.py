import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Your data
data = """
S10A8@42 ENOA@357 1433Z@94 CXCL7@89 PLSL@101 FLNA@1018 ITGA2B@96 GDIR2@76 CXCL7@105 TLN1@732 ITGA2B@87 ALDOA@339 ITGA2B@718 ITGA2B@161 CAP1@93 HBB@94 ENOA@389 A0A024R5Z9@49 TLN1@956 Q6FHU2@153 D3DPU2@427 G3P@247 CAP1@427 GELS@331 ACTN1@180 PF4V@70 HS71A@306 DEF1@83 TLN1@1023 ALDOA@178 TLN1@1927 PPIA@161 COR1A@332 TBA4A@347 TBB2A@12 PRDX5@100 PLSL@460 PPIA@62 PGK1@108 FLNA@2293
"""

ranks = """
41.443 21.271 13.153 44.505 32.562 91.958 25.446 8.105 41.641 94.441 32.579 38.958 22.696 34.825 33.085 58.237 67.509 7.073 73.948 50.523 30.624 1.679 30.624 29.848 55.145 72.263 9.458 1.235 29.599 2.485 33.795 86.054 12.591 26.619 30.458 20.121 28.76 2.384 87.705 24.395
"""

affinities = """
26785.55 12244.59 7992.59 27018.39 20011.18 37601.39 18008.07 2348.53 26564.75 35227.23 18365.01 17887.47 16629.09 24725.6 21044.43 25897.54 26295.95 5228.36 32843.04 29160.94 16642.77 642.04 16642.77 15967.71 28917.44 33191.34 5006.64 241.67 13184.29 895.22 21378.8 32759.27 11438.12 18390.86 18904.92 14577.78 19376.46 810.33 29093.82 19617.57
"""

# Split the strings into lists
data_list = data.split()
ranks_list = list(map(float, ranks.split()))
affinities_list = list(map(float, affinities.split()))

# Create a DataFrame
df = pd.DataFrame(list(zip(data_list, ranks_list, affinities_list)), columns=['Data', 'Rank', 'Affinity'])

# Apply T-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(df[['Rank', 'Affinity']])

# Plot the T-SNE result
plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])

# Annotate points with the corresponding data labels
for i, txt in enumerate(df['Data']):
    plt.annotate(txt, (tsne_result[i, 0], tsne_result[i, 1]))

plt.title('T-SNE Plot')
plt.xlabel('%Rank-EL+%Rank-BA')
plt.ylabel('Affinity')
# plt.show()

plt.savefig('wkf.svg', dpi=1200, bbox_inches='tight')
