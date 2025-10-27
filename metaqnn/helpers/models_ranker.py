import pandas as pd

def best_model_updater(dataframe, idx: int, best_models: list, cache_limit: int) -> list:
    current_model_score = (dataframe.iloc[idx]['accuracy'] - 0.25)**3 / dataframe.iloc[idx]['trainable_parameters']**0.25

    best_models.append([dataframe.iloc[idx]['ix_q_value_update'], current_model_score])
    best_models = sorted(best_models, key=lambda best_models: best_models[1])
    if len(best_models) > cache_limit:
        best_models.pop(0)

    return best_models


def model_ranker(file_path, cache_limit: int):
    ##2D arrays storing (model_index, score)
    best_models = []
    df = pd.read_csv(file_path)
    
    for i in range(len(df)):
        best_models = best_model_updater(df, i, best_models, cache_limit)

    return best_models

    # print([model[0] for model in best_models])

# model_ranker('./metaqnn/learner_logs/replay_database.csv', 10)