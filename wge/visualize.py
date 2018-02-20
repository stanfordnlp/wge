import gtd.ml.training_run_viewer
from gtd.ml.training_run_viewer import run_name, Commit, JSONSelector, NumSteps
from wge.training_run import MiniWoBTrainingRuns


class TrainingRunViewer(gtd.ml.training_run_viewer.TrainingRunViewer):

    def __init__(self):
        runs = MiniWoBTrainingRuns(check_commit=False)
        super(TrainingRunViewer, self).__init__(runs)

        metadata = lambda keys: JSONSelector('metadata.txt', keys)

        self.add('name', run_name)
        self.add('commit', Commit(), lambda s: s[:8])
        self.add('dataset', metadata(['config', 'dataset', 'path']))
        self.add('steps', NumSteps())
        self.add('host', metadata(['host']), lambda s: s[:10])
        self.add('last seen', metadata(['last_seen']))

        two_decimal = lambda f: '{:.2f}'.format(f)
