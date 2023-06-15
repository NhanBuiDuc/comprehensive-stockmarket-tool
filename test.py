from itertools import combinations

original_ensembled_model = {
    "svm": 1,
    "random_forest": 1,
    "xgboost": 2,
    "lstm": 2,
    "news": 1
}

keys = list(original_ensembled_model.keys())

combinations_list = []
for r in range(2, len(keys) + 1):
    combinations_list.extend(combinations(keys, r))

for combination in combinations_list:
    new_ensembled_model = original_ensembled_model.copy()
    for word in keys:
        if word not in combination:
            new_ensembled_model[word] = -1

    if list(new_ensembled_model.values()).count(-1) <= 3:
        print(new_ensembled_model)
