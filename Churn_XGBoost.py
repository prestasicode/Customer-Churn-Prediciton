
import pandas
import joblib, pickle
from sklearn.preprocessing import OneHotEncoder
import settings

def preprocessing(df, ohe_loc = None):
    encode_cols = df.loc[:, df.dtypes == "object"].columns.tolist()
    
    with open(ohe_loc, "rb") as ohe_file:
        ohe = pickle.load(ohe_file) 
    enc = ohe.transform(df[encode_cols])
    df[ohe.get_feature_names(encode_cols).tolist()] = pandas.DataFrame(enc, index=df.index)

    df.drop(encode_cols,1,inplace=True)

    df.dropna(inplace=True)
    
    return df

model_ = None
def scoreModel(avg_arpu_3m, acct_age, billing_cycle, credit_class, sales_channel, Est_HH_Income, region, cs_med_home_value, cs_pct_home_owner, forecast_region, handset, data_device_age, equip_age, delinq_indicator, avg_days_susp, calls_in_pk, calls_in_offpk, calls_out_offpk, calls_out_pk, mou_total_pct_MOM, mou_onnet_pct_MOM, mou_roam_pct_MOM, mou_onnet_6m_normal, mou_roam_6m_normal, voice_tot_bill_mou_curr, tot_voice_chrgs_curr, bill_data_usg_m03, bill_data_usg_m06, bill_data_usg_m09, mb_data_usg_m01, mb_data_usg_m02, mb_data_usg_m03, mb_data_ndist_mo6m, mb_data_usg_roamm01, mb_data_usg_roamm02, mb_data_usg_roamm03, data_usage_amt, data_prem_chrgs_curr, nbr_data_cdrs, avg_data_chrgs_3m, avg_data_prem_chrgs_3m, avg_overage_chrgs_3m, calls_TS_acct, wrk_orders, days_openwrkorders, calls_care_acct, calls_care_3mavg_acct, calls_care_6mavg_acct, res_calls_3mavg_acct, res_calls_6mavg_acct, curr_days_susp, pymts_late_ltd, calls_care_ltd, MB_Data_Usg_M04, MB_Data_Usg_M05, MB_Data_Usg_M06, MB_Data_Usg_M07, MB_Data_Usg_M08, MB_Data_Usg_M09, nbr_contacts):
    "Output: P_churn1, P_churn0, P_churn, msg"
    
    global model_
    try:
        if model_ is None:
            model_ = joblib.load(settings.pickle_path + "Churn_XGBoost.joblib")
        
        input_array = pandas.DataFrame([[avg_arpu_3m, acct_age, billing_cycle, credit_class, sales_channel, Est_HH_Income, region, cs_med_home_value, cs_pct_home_owner, forecast_region, handset, data_device_age, equip_age, delinq_indicator, avg_days_susp, calls_in_pk, calls_in_offpk, calls_out_offpk, calls_out_pk, mou_total_pct_MOM, mou_onnet_pct_MOM, mou_roam_pct_MOM, mou_onnet_6m_normal, mou_roam_6m_normal, voice_tot_bill_mou_curr, tot_voice_chrgs_curr, bill_data_usg_m03, bill_data_usg_m06, bill_data_usg_m09, mb_data_usg_m01, mb_data_usg_m02, mb_data_usg_m03, mb_data_ndist_mo6m, mb_data_usg_roamm01, mb_data_usg_roamm02, mb_data_usg_roamm03, data_usage_amt, data_prem_chrgs_curr, nbr_data_cdrs, avg_data_chrgs_3m, avg_data_prem_chrgs_3m, avg_overage_chrgs_3m, calls_TS_acct, wrk_orders, days_openwrkorders, calls_care_acct, calls_care_3mavg_acct, calls_care_6mavg_acct, res_calls_3mavg_acct, res_calls_6mavg_acct, curr_days_susp, pymts_late_ltd, calls_care_ltd, MB_Data_Usg_M04, MB_Data_Usg_M05, MB_Data_Usg_M06, MB_Data_Usg_M07, MB_Data_Usg_M08, MB_Data_Usg_M09, nbr_contacts]],
                                   columns = ["avg_arpu_3m", "acct_age", "billing_cycle", "credit_class", "sales_channel", "Est_HH_Income", "region", "cs_med_home_value", "cs_pct_home_owner", "forecast_region", "handset", "data_device_age", "equip_age", "delinq_indicator", "avg_days_susp", "calls_in_pk", "calls_in_offpk", "calls_out_offpk", "calls_out_pk", "mou_total_pct_MOM", "mou_onnet_pct_MOM", "mou_roam_pct_MOM", "mou_onnet_6m_normal", "mou_roam_6m_normal", "voice_tot_bill_mou_curr", "tot_voice_chrgs_curr", "bill_data_usg_m03", "bill_data_usg_m06", "bill_data_usg_m09", "mb_data_usg_m01", "mb_data_usg_m02", "mb_data_usg_m03", "mb_data_ndist_mo6m", "mb_data_usg_roamm01", "mb_data_usg_roamm02", "mb_data_usg_roamm03", "data_usage_amt", "data_prem_chrgs_curr", "nbr_data_cdrs", "avg_data_chrgs_3m", "avg_data_prem_chrgs_3m", "avg_overage_chrgs_3m", "calls_TS_acct", "wrk_orders", "days_openwrkorders", "calls_care_acct", "calls_care_3mavg_acct", "calls_care_6mavg_acct", "res_calls_3mavg_acct", "res_calls_6mavg_acct", "curr_days_susp", "pymts_late_ltd", "calls_care_ltd", "MB_Data_Usg_M04", "MB_Data_Usg_M05", "MB_Data_Usg_M06", "MB_Data_Usg_M07", "MB_Data_Usg_M08", "MB_Data_Usg_M09", "nbr_contacts"])
        input_array.fillna(0, inplace=True)
        # custom preprocessing step
        input_array = preprocessing(input_array, ohe_loc = settings.pickle_path + "Churn_encoder.pickle")
        result = model_.predict_proba(input_array)

        P_churn1, P_churn0, P_churn = float(result.tolist()[0][1]), float(result.tolist()[0][0]), float(round(result.tolist()[0][1],0)) 
        msg = "Not likely to churn" if P_churn==0 else "ALERT: Likely to churn."
        return P_churn1, P_churn0, str(P_churn), msg
    except Exception as e:
        return -1, -1, str(-1), str(e)