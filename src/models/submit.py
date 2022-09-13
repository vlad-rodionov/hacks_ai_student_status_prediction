from catboost import CatBoostClassifier
import pandas as pd
import click

SEED = 12345

CAT_FEATURES = [
    "Код_группы",
    "Один_знак",
    "Два_знака",
    "Три_знака",
    "Четыре_знака",
    "Пол",
    "Основания",
    "Изучаемый_Язык",
    "Страна_ПП",
    "Общежитие",
    "Наличие_Матери",
    "Наличие_Отца",
    "Страна_Родители",
    "Село",
    "Иностранец",
    "КодФакультета",
    "Сироты",
    "Полная_семья",
    "Группа_бал",
    "Тип_заведения",
    "Место_заведения",
    "Регион_ПП",
    "Город_ПП",
    "Возрастная_группа",
]

PARAMS = {
    "objective": "MultiClass",
    "colsample_bylevel": 0.09193906043107654,
    "depth": 10,
    "boosting_type": "Ordered",
    "bootstrap_type": "MVS",
    "learning_rate": 0.05,
    "n_estimators": 1404,
    "max_bin": 224,
    "min_data_in_leaf": 128,
    "l2_leaf_reg": 0.04355511389301864,
    "verbose": True,
    "random_state": SEED,
    "cat_features": CAT_FEATURES,
    "auto_class_weights": "Balanced",
    "eval_metric": "TotalF1:average=Macro",
}


@click.command()
@click.argument("input_path_train", type=click.Path())
@click.argument("input_path_test", type=click.Path())
@click.argument("input_path_submission", type=click.Path())
@click.argument("output_path_submission", type=click.Path())
def submit(
    input_path_train: str,
    input_path_test: str,
    input_path_submission: str,
    output_path_submission: str,
):

    data = pd.read_csv(input_path_train)
    test = pd.read_csv(input_path_test)
    submission = pd.read_csv(input_path_submission)

    features, target = data.drop(columns=["Статус"]), data["Статус"]

    model = CatBoostClassifier(**PARAMS)
    model.fit(features, target)

    predict_model = model.predict(test)

    submission["Статус"] = predict_model

    submission.to_csv(output_path_submission, index=False)


if __name__ == "main":
    submit()
