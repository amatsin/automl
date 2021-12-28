import json
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def inspect_trials(trials, baseline_loss, fn_name):
    print(f'There were {len(trials)} trials made.')
    check_if_better_than_baseline(trials, baseline_loss)
    filename = f'{len(trials)}_trials_for_{fn_name}_at_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    save_trials(trials, filename)
    save_trials_losses_graph(trials, filename)


def check_if_better_than_baseline(trials, baseline_loss):
    reached = False
    n_trial = 0
    for t in trials.trials:
        trial_loss = t['result']['loss']
        n_trial = t['tid'] + 1
        if not reached and trial_loss < baseline_loss:
            print(f'Reached a smaller loss {trial_loss} than baseline {baseline_loss} in {n_trial} trials')
            reached = True
    if not reached:
        print(f'Was not able to reach a smaller loss than baseline ({baseline_loss}) in {n_trial} trials')


def save_trials(trials, filename):
    with open(filename + '.json', mode='w') as fp:
        json.dump(trials.trials, fp, indent=4, sort_keys=True, default=str)


def save_trials_losses_graph(trials, filename):
    f, ax = plt.subplots(1)
    xs = [t['tid'] for t in trials.trials]
    ys = [t['result']['loss'] for t in trials.trials]
    ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('$loss$ $vs$ $trial$ ', fontsize=18)
    ax.set_xlabel('$trial$', fontsize=16)
    ax.set_ylabel('$loss$', fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(filename + '.png')
