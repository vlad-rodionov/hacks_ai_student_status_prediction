import pandas as pd
from catboost import CatBoostClassifier
import click


@click.command()
@click.argument("input_path_model", type=click.Path())
@click.argument("input_path_test", type=click.Path())
@click.argument("input_path_submit", type=click.Path())
@click.argument("output_path_submit", type=click.Path())
def submit(
    input_path_model: str,
    input_path_test: str,
    input_path_submit: str,
    output_path_submit: str,
):

    test = pd.read_csv(input_path_test)
    submission = pd.read_csv(input_path_submit)

    model = CatBoostClassifier()
    model.load_model(input_path_model)

    predict_test = model.predict(test)

    submission["Статус"] = predict_test

    submission.to_csv(output_path_submit, index=False)


if __name__ == "main":
    submit()
