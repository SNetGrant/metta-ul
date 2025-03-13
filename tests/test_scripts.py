from hyperon import MeTTa, E
from pathlib import Path
# from skl import generator
pwd = Path(__file__).parent


def process_exceptions(results):
    for result in results:
        assert result in [[E()], []]


def run_script(fname):
    with open(fname) as f:
        return MeTTa().run(f.read())


def test_scripts():
    process_exceptions(run_script(f"{pwd}/pdm_test.metta"))
    process_exceptions(run_script(f"{pwd}/norm_test.metta"))
    process_exceptions(run_script(f"{pwd}/kmeans_test.metta"))
    process_exceptions(run_script(f"{pwd}/plot_test.metta"))
    return
