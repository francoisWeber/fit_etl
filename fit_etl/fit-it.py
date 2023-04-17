import click

from fit_etl.extract import Workout
from fit_etl.analysis import BreakAnalyzer
from fit_etl.tooling import constants as cst


@click.command()
@click.argument(
    "filename", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.option("--dont-smooth", is_flag=True, default=False)
def run(filename, dont_smooth):
    use_smoothed_version = not dont_smooth
    if filename.endswith(".fit"):
        workout = Workout.from_fit(filename)
    else:
        raise NotImplementedError("No reader for this file type")

    break_analyzer = BreakAnalyzer(use_smoothed_versions=use_smoothed_version)
    a = workout.get_ith_lap_of_category(0, cst.LAP_CATEGORY_BREAK)
    break_analyzer.fit(a)
    print(break_analyzer.get_key_params())


if __name__ == "__main__":
    run()
