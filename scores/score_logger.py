from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np

SCORES_CSV_PATH = "./scores/scores.csv"
SCORES_PNG_PATH = "./scores/scores.png"
SOLVED_CSV_PATH = "./scores/solved.csv"
SOLVED_PNG_PATH = "./scores/solved.png"
AVERAGE_SCORE_TO_SOLVE = 195
CONSECUTIVE_RUNS_TO_SOLVE = 100


class ScoreLogger:

    def __init__(self, env_name):
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.env_name = env_name

        if os.path.exists(SCORES_PNG_PATH):
            os.remove(SCORES_PNG_PATH)
        if os.path.exists(SCORES_CSV_PATH):
            os.remove(SCORES_CSV_PATH)

    def add_score(self, score, run):
        self._save_csv(SCORES_CSV_PATH, score)
        self._save_png(input_path=SCORES_CSV_PATH,
                       output_path=SCORES_PNG_PATH,
                       x_label="runs",
                       y_label="scores",
                       average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
                       show_goal=True,
                       show_trend=True,
                       show_legend=True)
        self.scores.append(score)
        mean_score = mean(self.scores)
        print("Scores: (min: " + str(min(self.scores)) + ", avg: " + str(mean_score) + ", max: " + str(max(self.scores)) + ")\n")
        if mean_score >= AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
            solve_score = run-CONSECUTIVE_RUNS_TO_SOLVE
            print("Solved in " + str(solve_score) + " runs, " + str(run) + " total runs.")
            self._save_csv(SOLVED_CSV_PATH, solve_score)
            self._save_png(input_path=SOLVED_CSV_PATH,
                           output_path=SOLVED_PNG_PATH,
                           x_label="trials",
                           y_label="steps before solve",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False)
            exit()

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.subplots()
        plt.figure().set_size_inches(8, 6)
        plt.plot(x, y, label="score per run")

        if type(x) != type(None) and show_trend and len(x) >= average_of_n_last * 2 :
            y_max_roll = self.max_rolling(np.array([y]), average_of_n_last, 2).tolist()[0]
            y_max_roll = self.avg_rolling(y_max_roll, average_of_n_last)
            y_min_roll = self.min_rolling(np.array([y]), average_of_n_last, 2).tolist()[0]
            y_min_roll = self.avg_rolling(y_min_roll, average_of_n_last)
            plt.plot(x[-len(y_max_roll):], y_max_roll, label="average max")
            plt.plot(x[-len(y_min_roll):], y_min_roll, label="average min")

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--", label="last " + str(average_range) + " runs average")

        if show_goal:
            plt.plot(x, [AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":", label=str(AVERAGE_SCORE_TO_SOLVE) + " score average goal")

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.cla()
        plt.figure().clear()
        plt.clf()
        plt.close('all')

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])

    # Source: https://stackoverflow.com/a/52219082/2847743
    def max_rolling(self, a, window, axis=1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.max(rolling,axis=axis)

    # Source: https://stackoverflow.com/a/52219082/2847743
    def min_rolling(self, a, window, axis=1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.min(rolling,axis=axis)

    # Source: https://stackoverflow.com/a/69950033/2847743
    def avg_rolling(self, a, n):
        N = len(a)
        return np.array([np.mean(a[i : i + n]) for i in np.arange(0, N - n + 1)])
