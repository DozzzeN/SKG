import numpy as np
from deap import base, creator, tools, algorithms

from segmentation.methods import dtw_metric


def evaluate(individual, segments, original_segments, segment_index):
    new_segment = list(original_segments[segment_index])
    for i, pos in enumerate(individual):
        new_segment[int(pos)] = np.random.uniform(-1, 2)

    new_dtw_distance = dtw_metric(new_segment, original_segments[segment_index])
    max_other_distances = max(dtw_metric(new_segment, seg) for i, seg in enumerate(segments) if i != segment_index)

    return new_dtw_distance, -max_other_distances


def modify_segments_position_compared_with_all_segments(segments, replace_number, seed):
    original_segments = [list(segment) for segment in segments]
    modified_segments = original_segments.copy()
    modifications = []
    np.random.seed(seed)

    for segment_index in range(len(segments)):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_int", np.random.randint, 0, len(original_segments[segment_index]))
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=replace_number)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(original_segments[segment_index]) - 1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", evaluate, segments=segments, original_segments=original_segments,
                         segment_index=segment_index)

        population = toolbox.population(n=50)
        algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.2, ngen=40, stats=None,
                                  halloffame=None, verbose=False)

        best_ind = tools.selBest(population, 1)[0]
        for pos in best_ind:
            new_value = np.random.uniform(-1, 2)
            modifications.append((segment_index, int(pos), new_value))
            modified_segments[segment_index][int(pos)] = new_value

    return modified_segments, modifications


# 测试代码
segments = [
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7]
]

replace_number = 2
seed = 42

modified_segments, modifications = modify_segments_position_compared_with_all_segments(segments, replace_number, seed)

print("Modified Segments:")
for segment in modified_segments:
    print(segment)

print("\nModifications:")
for mod in modifications:
    print(mod)
