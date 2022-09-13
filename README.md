# hacks_ai_student_status_prediction
[Ссылка на соревнование](https://hacks-ai.ru/championships/758263)

## Цель и результат соревнования

Соревнование запущено Алтайским государственным университетом и компанией Thrive Technologies LLC. 
Основная задача чемпионата - предсказать будущий 
потенциальный статус студента на основе данных нескольких 
тысяч студентов. Основной метрикой чемпионата выбран F1_score.

Базовое решение, представленное авторами соревнования, представляло собой RandomForestClassifier лес с и
тоговой метрикой 0.7 на публичном лидерборде.

С помощью CatBoostClassifier и дополнительным Feature Engineering над датасетом удалось достичь метрики 0.798 на публичном
лидерборде и занять 4 место. В текущий момент приватный лидерборд еще не открыт.

### Requirements
- Python 3
- NumPy
- Pandas
- Catboost
- Sklearn
- Optuna

## Решение
Итоговым решением является CatBoostClassifier, гиперпараметры которого найдены с помощью пакета Optuna.
Локальная валидация происходила как с помощью 5-fold CV, так и с помощью hold-out выборки.

Финальная модель была обучена на всем датасете.

## EDA
С предварительным анализом можно ознакомиться [здесь](https://github.com/vlad-rodionov/hacks_ai_student_status_prediction/blob/main/Notebooks/EDA_student.ipynb)

## Features

Важность фичей, полученная встроенными методами Catboost:

![feature_importance](https://github.com/vlad-rodionov/hacks_ai_student_status_prediction/blob/main/reports/feature_importance.png)

Важность фичей, полученная с помощью библиотеки SHAP:

![SHAP_values](https://github.com/vlad-rodionov/hacks_ai_student_status_prediction/blob/main/reports/SHAP.png)

Заметим, что половина из ТОП-10 фич - это фичи, которые были сделаны "руками"