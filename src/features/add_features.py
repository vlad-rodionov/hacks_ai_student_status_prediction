import pandas as pd
import numpy as np
import click


@click.command()
@click.argument("input_path_train", type=click.Path())
@click.argument("input_path_test", type=click.Path())
@click.argument("output_path_train", type=click.Path())
@click.argument("output_path_test", type=click.Path())
def add_features(
    input_path_train: str,
    input_path_test: str,
    output_path_train: str,
    output_path_test: str
):
    """
    :param input_path_train: Path to read cleaned Train DataFrame
    :param output_path_train: Path to save Train DataFrame with handle features
    :param input_path_test: Path to read cleaned Test DataFrame
    :param output_path_test: Path to save Test DataFrame with handle features
    :return:
    """

    clean_data_train = pd.read_csv(input_path_train)
    clean_data_test = pd.read_csv(input_path_test)

    # Длина наименования учебного заведения
    clean_data_train["Длина_УЗ"] = clean_data_train["Уч_Заведение"].apply(
        lambda val: len(val)
    )
    clean_data_test["Длина_УЗ"] = clean_data_test["Уч_Заведение"].apply(
        lambda val: len(val)
    )

    # Длина группы
    def _return_digits(n, i):
        j = 0
        while j < i:
            cur, d = divmod(n, 10)
            n = cur
            j += 1

        return n

    clean_data_train["Один_знак"] = clean_data_train["Код_группы"].apply(
        _return_digits, i=4
    )
    clean_data_train["Два_знака"] = clean_data_train["Код_группы"].apply(
        _return_digits, i=3
    )
    clean_data_train["Три_знака"] = clean_data_train["Код_группы"].apply(
        _return_digits, i=2
    )
    clean_data_train["Четыре_знака"] = clean_data_train["Код_группы"].apply(
        _return_digits, i=1
    )

    clean_data_test["Один_знак"] = clean_data_test["Код_группы"].apply(
        _return_digits, i=4
    )
    clean_data_test["Два_знака"] = clean_data_test["Код_группы"].apply(
        _return_digits, i=3
    )
    clean_data_test["Три_знака"] = clean_data_test["Код_группы"].apply(
        _return_digits, i=2
    )
    clean_data_test["Четыре_знака"] = clean_data_test["Код_группы"].apply(
        _return_digits, i=1
    )

    # Приведение Уч_Заведение к единому типу
    def _return_edu_institution(value):

        value = value.lower()

        if "университет" in value:
            return 1
        if "школа" in value:
            return 2
        if "лицей" in value:
            return 3
        if "колледж" in value:
            return 4
        if "институт" in value:
            return 5
        if "академия" in value:
            return 6
        if "гимназия" in value:
            return 7
        if "техникум" in value:
            return 8
        if "сош" in value:
            return 2

        return 9

    clean_data_train["Тип_заведения"] = clean_data_train["Уч_Заведение"].apply(
        _return_edu_institution
    )
    clean_data_test["Тип_заведения"] = clean_data_test["Уч_Заведение"].apply(
        _return_edu_institution
    )
    clean_data_train["Тип_заведения"] = clean_data_train["Тип_заведения"].astype(
        np.int8
    )
    clean_data_test["Тип_заведения"] = clean_data_test["Тип_заведения"].astype(np.int8)

    # Признаки на дате рождения
    clean_data_train["Поступление_рождение"] = (
        clean_data_train["Год_Поступления"] - clean_data_train["Дата_Рождения"]
    )
    clean_data_test["Поступление_рождение"] = (
        clean_data_test["Год_Поступления"] - clean_data_test["Дата_Рождения"]
    )

    clean_data_train["Выпуск_поступление"] = (
        clean_data_train["Год_Окончания_УЗ"] - clean_data_train["Год_Поступления"]
    )
    clean_data_test["Выпуск_поступление"] = (
        clean_data_test["Год_Окончания_УЗ"] - clean_data_test["Год_Поступления"]
    )

    # СрБалАттестата
    clean_data_train["СрБаллАттестата"] = clean_data_train["СрБаллАттестата"].apply(
        lambda score: score if score > 5 else score * 17
    )
    clean_data_test["СрБаллАттестата"] = clean_data_test["СрБаллАттестата"].apply(
        lambda score: score if score > 5 else score * 17
    )

    # Информация о семье
    clean_data_train["Полная_семья"] = (
        clean_data_train["Наличие_Матери"] & clean_data_train["Наличие_Отца"]
    )
    clean_data_test["Полная_семья"] = (
        clean_data_test["Наличие_Матери"] & clean_data_test["Наличие_Отца"]
    )
    clean_data_train["Полная_семья"] = clean_data_train["Полная_семья"].astype(np.int8)
    clean_data_test["Полная_семья"] = clean_data_test["Полная_семья"].astype(np.int8)

    clean_data_train["Сироты"] = (
        (clean_data_train["Наличие_Отца"] == 0)
        & (clean_data_train["Наличие_Матери"] == 0)
    ) * 1
    clean_data_test["Сироты"] = (
        (clean_data_test["Наличие_Отца"] == 0)
        & (clean_data_test["Наличие_Матери"] == 0)
    ) * 1
    clean_data_train["Сироты"] = clean_data_train["Сироты"].astype(np.int8)
    clean_data_test["Сироты"] = clean_data_test["Сироты"].astype(np.int8)

    # Информация об аттестатах
    def _return_gpa(value):

        if value < 30:
            return 1  # Очень низкий бал
        if value < 50:
            return 2  # Низкий бал
        if value < 70:
            return 3  # Норм
        if value < 80:
            return 4  # Высокий
        if value < 90:
            return 5  # Очень высокий
        return 6  # Очень-очень высокий

    clean_data_train["Группа_бал"] = clean_data_train["СрБаллАттестата"].apply(
        _return_gpa
    )
    clean_data_test["Группа_бал"] = clean_data_test["СрБаллАттестата"].apply(
        _return_gpa
    )
    clean_data_train["Группа_бал"] = clean_data_train["Группа_бал"].astype(np.int8)
    clean_data_test["Группа_бал"] = clean_data_test["Группа_бал"].astype(np.int8)

    # Где_Находится_УЗ
    def _return_place_institution(value):

        value = value.lower()

        if "барнаул" in value:
            return 1
        if "бийск" in value:
            return 2
        if "москва" in value:
            return 3
        if "санкт-перетбург" in value:
            return 4
        if "казахстан" in value:
            return 5
        if "алтай" in value:
            return 6

        return 9

    clean_data_train["Место_заведения"] = clean_data_train["Где_Находится_УЗ"].apply(
        _return_place_institution
    )
    clean_data_test["Место_заведения"] = clean_data_test["Где_Находится_УЗ"].apply(
        _return_place_institution
    )
    clean_data_train["Место_заведения"] = clean_data_train["Место_заведения"].astype(
        np.int8
    )
    clean_data_test["Место_заведения"] = clean_data_test["Место_заведения"].astype(
        np.int8
    )

    # Возрастная_группа
    def _return_age_group(value):

        if value < 1960:
            return 1
        if value < 1970:
            return 2
        if value < 1980:
            return 3
        if value < 1990:
            return 4
        if value < 2000:
            return 5

        return 6

    clean_data_train["Возрастная_группа"] = clean_data_train["Дата_Рождения"].apply(
        _return_age_group
    )
    clean_data_test["Возрастная_группа"] = clean_data_test["Дата_Рождения"].apply(
        _return_age_group
    )
    clean_data_train["Возрастная_группа"] = clean_data_train[
        "Возрастная_группа"
    ].astype(np.int8)
    clean_data_test["Возрастная_группа"] = clean_data_test["Возрастная_группа"].astype(
        np.int8
    )

    clean_data_train = clean_data_train.drop(
        columns=["Уч_Заведение", "Где_Находится_УЗ"]
    )
    clean_data_test = clean_data_test.drop(columns=["Уч_Заведение", "Где_Находится_УЗ"])

    clean_data_train.to_csv(output_path_train, index=False)
    clean_data_test.to_csv(output_path_test, index=False)


if __name__ == "__main__":
    add_features()
