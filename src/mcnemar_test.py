import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def create_contingency_table(true_pred, baseline_pred, optimized_pred):
    mc_data = pd.DataFrame()
    mc_data["true_pred"] = true_pred
    mc_data["baseline_pred"] = baseline_pred
    mc_data["optimized_pred"] = optimized_pred

    mc_data.loc[mc_data['true_pred'] == mc_data['baseline_pred'], 'correct_baseline'] = "Yes"
    mc_data.loc[mc_data['true_pred'] == mc_data['optimized_pred'], 'correct_optimized'] = "Yes"
    mc_data.loc[mc_data['true_pred'] != mc_data['baseline_pred'], 'correct_baseline'] = "No"
    mc_data.loc[mc_data['true_pred'] != mc_data['optimized_pred'], 'correct_optimized'] = "No"

    contingency_table_df = pd.DataFrame(data={
        "nr_correct_baseline": ["Yes/Yes", "No/Yes"],
        "nr_incorrect_baseline": ["Yes/No", "No/No"]}, index=["nr_correct_optimized", "nr_incorrect_optimized"])
    nr_corr_base_corr_opt = 0
    nr_corr_base_incorr_opt = 0
    nr_incorr_base_corr_opt = 0
    nr_incorr_base_incorr_opt = 0
    for index, row in mc_data.iterrows():
        if row['correct_baseline'] == "Yes" and row['correct_optimized'] == "Yes":
            nr_corr_base_corr_opt += 1
        elif row['correct_baseline'] == "Yes" and row['correct_optimized'] == "No":
            nr_corr_base_incorr_opt += 1
        elif row['correct_baseline'] == "No" and row['correct_optimized'] == "Yes":
            nr_incorr_base_corr_opt += 1
        elif row['correct_baseline'] == "No" and row['correct_optimized'] == "No":
            nr_incorr_base_incorr_opt += 1

    contingency_table_df.iloc[0, 0] = nr_corr_base_corr_opt
    contingency_table_df.iloc[0, 1] = nr_corr_base_incorr_opt
    contingency_table_df.iloc[1, 0] = nr_incorr_base_corr_opt
    contingency_table_df.iloc[1, 1] = nr_incorr_base_incorr_opt
    return contingency_table_df


def calculate_mcnemar_test(contingency_df):
    # calculate mcnemar test
    result = mcnemar(contingency_df.to_numpy(), exact=False, correction=True)
    # summarize the finding
    print('statistic=%.5f, p-value=%.20f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')