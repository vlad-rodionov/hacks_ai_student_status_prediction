import pandas as pd
import numpy as np
import click

MODE_COLS = [
    "Пол",
    "Изучаемый_Язык",
    "Уч_Заведение",
    "Где_Находится_УЗ",
    "Год_Окончания_УЗ",
    "Страна_ПП",
    "Регион_ПП",
    "Город_ПП",
    "Общежитие",
    "Страна_Родители",
    "Село",
    "Иностранец",
]

INT_COLS = [
    "Год_Поступления",
    "Пол",
    "Основания",
    "Изучаемый_Язык",
    "Дата_Рождения",
    "Год_Окончания_УЗ",
    "Общежитие",
    "Наличие_Матери",
    "Наличие_Отца",
    "Село",
    "Иностранец",
    "КодФакультета",
    "СрБаллАттестата",
]

OBJECT_COLS = [
    "Уч_Заведение",
    "Где_Находится_УЗ",
    "Страна_ПП",
    "Регион_ПП",
    "Город_ПП",
    "Страна_Родители",
]


@click.command()
@click.argument("input_path_train", type=click.Path())
@click.argument("output_path_train", type=click.Path())
@click.argument("input_path_test", type=click.Path())
@click.argument("output_path_test", type=click.Path())
def clean_data(
        input_path_train: str,
        output_path_train: str,
        input_path_test: str,
        output_path_test: str,
):
    """
    :param input_path_train: Path to read Train DataFrame
    :param output_path_train: Path to save cleaned Train DataFrame
    :param input_path_test: Path to read Test DataFrame
    :param output_path_test: Path to save cleaned Test DataFrame
    :return:
    """

    data_train = pd.read_csv(input_path_train)
    data_test = pd.read_csv(input_path_test)

    for col in MODE_COLS:
        filled_value = data_train[col].value_counts().nlargest(1).index[0]
        data_train[col] = data_train[col].fillna(filled_value)
        data_test[col] = data_test[col].fillna(filled_value)

    delete_admission_index = data_train[
        data_train["Год_Поступления"] > 2020
        ].index
    data_train = data_train.drop(labels=delete_admission_index, axis=0)

    sex_map = {"Жен": 1.0, "Муж": 0.0, "муж": 0.0, "жен": 1.0}
    data_train["Пол"] = data_train["Пол"].map(sex_map)
    data_test["Пол"] = data_test["Пол"].map(sex_map)

    reason_map = {"СН": 1, "ЦН": 2, "БН": 3, "ОО": 4, "ДН": 5, "ЛН": 4}
    data_train["Основания"] = data_train["Основания"].map(reason_map)
    data_test["Основания"] = data_test["Основания"].map(reason_map)

    language_map = {
        "Английский язык": 1,
        "Немецкий язык": 2,
        "Французский язык": 3,
        "Англиийский": 1,
        "Иностранный язык (Английский)": 1,
        "Русский язык": 4,
        "Английский, немецкий языки": 1,
        "Иностранный язык (Немецкий)": 2,
    }

    data_train["Изучаемый_Язык"] = data_train["Изучаемый_Язык"].map(language_map)
    data_test["Изучаемый_Язык"] = data_test["Изучаемый_Язык"].map(language_map)

    data_train["Дата_Рождения"] = pd.to_datetime(data_train["Дата_Рождения"]).apply(
        lambda date: date.year
    )
    data_test["Дата_Рождения"] = pd.to_datetime(data_test["Дата_Рождения"]).apply(
        lambda date: date.year
    )

    delete_birth_index = data_train[data_train["Дата_Рождения"] > 2004].index
    data_train = data_train.drop(labels=delete_birth_index, axis=0)

    delete_gpa_index = data_train[data_train["СрБаллАттестата"] > 100].index
    data_train = data_train.drop(labels=delete_gpa_index, axis=0)

    data_train[INT_COLS] = data_train[INT_COLS].astype(np.int8)
    data_test[INT_COLS] = data_test[INT_COLS].astype(np.int8)

    data_train[OBJECT_COLS] = data_train[OBJECT_COLS].astype("object")
    data_test[OBJECT_COLS] = data_test[OBJECT_COLS].astype("object")

    data_train.to_csv(output_path_train)
    data_test.to_csv(output_path_test)


if __name__ == "__main__":
    clean_data()
