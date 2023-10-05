# import python librairies
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# load json files with results
def json_results_to_df(list_models):

    list_df = []

    for i in range(len(list_models)):
        result_json = f'/workspace/results/result_model_{list_models[i]}.json'

        # load json file
        with open(result_json) as json_file:
            result_data = json.load(json_file)

        # transform it into dataframe
        globals()[f'dataframe_result_{list_models[i]}'] = pd.DataFrame.from_dict(result_data)
        df_sentiment = globals()[f'dataframe_result_{list_models[i]}'][[f"rate_{list_models[i]}"]]
        list_df.append(df_sentiment)

    return list_df


# calculate metrics
def metrics_calculation_sentiment(result_sentiment, list_models):

    metrics_matrix = {}
    metrics_matrix['metric'] = ["accuracy", "precision", "recall", "f1 score"]

    y_true = result_sentiment[['rating']]

    for model in range(len(list_models)):

        metrics_value = []

        # ex: y_pred_a = df[['stars_lettria']]
        y_pred = result_sentiment[[f"rate_{list_models[model]}"]]

        # evaluation is done for each model
        print(f"Metrics calculation for model {list_models[model]}:\n")

        confusion = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix\n', confusion)

        print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))

        print('Macro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='macro')))
        print('Macro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='macro')))
        print('Macro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='macro')))

        metrics_value.extend((accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average='macro'), \
                            recall_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='macro')))
        metrics_matrix[f"model_{list_models[model]}"] = metrics_value

    metrics_df = pd.DataFrame(metrics_matrix)
    metrics_df.set_index('metric')
    print(metrics_df)

    return metrics_df


# main
if __name__ == '__main__':
    
    # load test set
    ground_truth = '/workspace/data/dataset_test.json'
    with open(ground_truth) as json_file:
        test_set = json.load(json_file)
        
    # define list of models
    list_models = ['lettria', 'bert', 'lstm']
    
    # json to dataframe
    df_results = json_results_to_df(list_models)
    
    # concat dataframes
    df_results_global = pd.concat([pd.DataFrame(test_set), df_results[0], df_results[1], df_results[2]], axis=1)
    
    # save the "global" results as json file
    df_results_global.to_json('/workspace/results/glogal_result_models.json')
    
    # calculate metrics
    metrics_comparaison = metrics_calculation_sentiment(df_results_global, list_models)
