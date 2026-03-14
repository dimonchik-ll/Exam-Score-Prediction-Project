from model import tune_models, fit_and_select_best, test_score, save_object

cv_scores, best_params = tune_models(n_trials=100)

best_model_name, val_score, best_pipeline = fit_and_select_best(best_params)

test_rmse = test_score(best_pipeline)

save_object(best_pipeline, "models_and_pipelines/final_pipeline.pkl")

print(f"Best model: {best_model_name}")
print(f"Test score: {test_rmse}")