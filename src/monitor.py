import json

import hyperopt.base
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


class Monitor:
    def __init__(self, baseline_loss, filename):
        self.baseline_loss = baseline_loss
        self.filename = filename
        self.reached_baseline = False
        self.smallest_loss = 1.0

    def early_stop_fn(self, trials, *args):
        last_result = trials.results[-1]
        if last_result['status'] != 'ok':
            return True, args
        if last_result['loss'] < self.baseline_loss:
            print(
                f'Reached better result {last_result["loss"]} than baseline {self.baseline_loss} in {len(trials.results)} trials')
            return False, args
        return False, args

    def inspect_result(self, trials: hyperopt.base.Trials, *args):
        if len(trials) % 10 != 0:
            return False, args
        self.save_best_hyperparams(trials)
        self.check_if_better_than_baseline(trials)
        self.save_trials(trials)
        self.save_trials_auc_graph(trials)
        return False, args

    def save_best_hyperparams(self, trials):
        best_loss = trials.best_trial['result']['loss']
        if best_loss >= self.smallest_loss:
            return
        self.smallest_loss = best_loss
        tid_ = trials.best_trial['tid']
        print(f'Trial number {tid_} with loss {best_loss} is the best one so far. Saving hyper parameters to file. '
               f'Params:')
        print(trials.argmin)
        with open(f'{self.filename}_trial_{tid_}_loss_{best_loss}.json', mode='w') as fp:
            json.dump(trials.argmin, fp, indent=4, sort_keys=True, default=str)

    def check_if_better_than_baseline(self, trials):
        if self.reached_baseline:
            return

        n_trial = len(trials)
        trial_loss = trials.best_trial['result']['loss']
        if trial_loss < self.baseline_loss:
            print(f'Reached a smaller loss {trial_loss} than baseline {self.baseline_loss} in {n_trial} trials')
            self.reached_baseline = True
        else:
            print(f'Was not able to reach a smaller loss than baseline ({self.baseline_loss}) in {n_trial} trials')

    def save_trials(self, trials):
        with open(self.filename + '.json', mode='w') as fp:
            json.dump(trials.trials, fp, indent=4, sort_keys=True, default=str)

    def save_trials_auc_graph(self, trials, title=None, y_lim=(0.55, 0.95)):
        f, ax = plt.subplots(1)
        try:
            xs = [t['tid'] for t in trials.trials]
            ys = [1 - t['result']['loss'] for t in trials.trials]
        except AttributeError:
            xs = [t['tid'] for t in trials]
            ys = [1 - t['result']['loss'] for t in trials]
        ax.set_xlim([0, len(trials)])
        ax.set_ylim(y_lim)
        ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, label='AUC at trial')
        plt.axhline(y=1 - self.baseline_loss, color='r', label='baseline')
        ax.legend()
        ax.set_title(title if title else 'AUC $vs$ $trial$ ', fontsize=18)
        ax.set_xlabel('$trial$', fontsize=16)
        ax.set_ylabel('$AUC$', fontsize=16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(self.filename + '.png')
