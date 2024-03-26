import csv
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_percentage_error


def fn_tam_concat_predict_m(r_path: str, f_month: str, res_dic: {}) -> {}:
    """
    input r_path : predict_13m이 있는 경로
    input res_dic : 모델의 전망 값 dictionary
    output : 해당 모델의 전망 값 dictionary(key : 시작 예측 연월)
    """
    res = pd.read_csv(f'{target_y_name}.txt', sep=" ", header=None).iloc[:, :-1].transpose()

    for i in range(len(res)):
        test_start_date = pd.to_datetime(f_month) + relativedelta(months=-len(res) + i)

        test_end_date = test_start_date + relativedelta(months=len(res) - 1)

        r = pd.date_range(start=test_start_date, end=test_end_date, freq="MS")

        key = (
            str(test_start_date).split("-")[0]
            + "-"
            + str(test_start_date).split("-")[1]
        )

        try:
            res_dic[key] = pd.concat(
                (res_dic[key], pd.DataFrame({"pred": res[i]})), axis=1
            )

        except:
            res_dic[key] = pd.DataFrame({"date": r, "pred": res[i]})

    return res_dic


def fn_tam_merge_y(tar_res_dic: {}, f_month) -> {}:
    """
    input param tar_res_dic : 타겟과 date로 이루어진 이중 딕셔너리
    """

    cutoff_date = str(pd.to_datetime(f_month) + relativedelta(months=-1))

    for target in tar_res_dic:
        if target in [
            "PRICE_COKE_GNR_GRN_ICC_RMB",
            "PRICE_AG_HIGH_ICC_RMB",
            "PRICE_SV_LIPF6_ICC_RMB",
        ]:
            data = f"./{target}_anly_data_{f_month}.csv"
        else:
            data = RAW_PATH
            # data = f"./INTEGR_METAL_NON_METAL_MONTHLY_{f_month}.csv"

        target_df = pd.read_csv(data).rename(columns={"Unnamed: 0": "date"})[
            ["date", target]
        ]
        target_df = target_df.loc[target_df.date <= cutoff_date]
        target_df["date"] = target_df["date"].astype("datetime64[ns]")

        for date in tar_res_dic[target]:
            tar_res_dic[target][date] = pd.merge(
                tar_res_dic[target][date], target_df, how="left"
            )

    return tar_res_dic


def fn_make_tam(r_path: str, f_month: str, target: str, res_path: str):
    """
    메인 함수

    코드를 돌리고 있는 경로 내에 "data/{target}_anly_data_{f_month}.csv" 메탈의 경우 f"data/INTEGR_METAL_NON_METAL_MONTHLY_{f_month}.csv"형식의 데이터가 있어야 합니다.


    input param r_path : 결과값이 저장된 경로(타겟, 예측 시작 월 값이 포함되어야함, 콜랩 폴더 경로 통일화 참조)
    input param f_month : 예측 시작 연월 형식 'yyyy-mm'
    input param target : 타겟 이름
    input param res_path : 저장하고자 하는 경로

    output : res_path 경로에 tam이 생김
    """

    res_dic = {}
    tar_res_dic = {}
    tar_res_dic[target] = fn_tam_concat_predict_m(r_path, f_month, res_dic)
    tar_res_dic = fn_tam_merge_y(tar_res_dic, f_month)
    tam_res_dic = tar_res_dic[target]
    first_key = list(tam_res_dic.keys())[0]
    horizon = len(tam_res_dic[first_key])

    f_dic = {}
    for d in tam_res_dic:
        tam_res_dic[d] = tam_res_dic[d].dropna(axis=0)
    for seq in tam_res_dic:
        for ahead in range(horizon):
            for avg in range(len(tam_res_dic[seq]) - ahead):
                if (
                    len(tam_res_dic[seq].loc[ahead : ahead + avg].dropna(axis=0))
                    < avg + 1
                ):
                    continue
                else:
                    try:
                        f_dic[f"{ahead + 1}_ahead_{avg + 1}_AVG"] = pd.concat(
                            [
                                f_dic[f"{ahead + 1}_ahead_{avg + 1}_AVG"],
                                tam_res_dic[seq].loc[ahead : ahead + avg],
                            ]
                        )

                    except:
                        f_dic[f"{ahead + 1}_ahead_{avg + 1}_AVG"] = tam_res_dic[
                            seq
                        ].loc[ahead : ahead + avg]
    tam_dic = {}
    for ahead_avg in f_dic:
        tam_dic[ahead_avg] = round(
            (
                1
                - mean_absolute_percentage_error(
                    f_dic[ahead_avg][target], f_dic[ahead_avg]["pred"]
                )
            )
            * 100,
            2,
        )
    csv_line = []
    for avg in range(horizon, 0, -1):
        avg_line = []
        for ahead in range(1, horizon + 1):
            try:
                avg_line.append(tam_dic[f"{ahead}_ahead_{avg}_AVG"])

            except:
                continue
        csv_line.append(avg_line)
    w = open(res_path, "w", encoding="utf-8", newline="")
    wr = csv.writer(w)
    for i in csv_line:
        wr.writerow(i)
    w.close()
