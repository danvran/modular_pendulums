"""
Copyright © 2023 Daniel Vranješ
You may use, distribute and modify this code under the MIT license.
You should have received a copy of the MIT license with this file.
If not, please visit https://github.com/danvran/modular_pendulums
"""

import itertools
import pandas
import numpy

class ExperimentHandler:
    def __init__(self, cross_val_data: list[any], exp_run_func: callable, metrics: list[str], *args: tuple[any, list[any]]):
        r"""
        param: cross_val_data: each entry represents the data for a cross validation run in the form as expexted by the exp_run_func
        param: exp_run_func: function to be executed as an experiment
        param: metrics: list with names of metrics as returned by the exp_run_func
        param: args: variables and possible values in the form (name, [value1, value2, ...])
        """
        self.parameter_names = []
        self.metric_names = metrics
        self.extended_metric_names = []
        for entry in self.metric_names:
            self.extended_metric_names.append(entry+"_mean")
            self.extended_metric_names.append(entry+"_std")
        self.cross_val_data = cross_val_data
        self.log_prefix = ['Exp']
        self.exp_run_func = exp_run_func
        self.parametersets = self.bundle(*args)
        self.log_columns = self.log_prefix + self.parameter_names + self.extended_metric_names
        self.log_df = pandas.DataFrame(columns=self.log_columns)
        self.combinations = self.make_parameter_combinations(self.parametersets)
        
    def bundle(self, *args: tuple[str, list[any]]) -> dict[any]:
        r"""
        description: takes arbitrary many tuples where the first entry represents a key and
        the second the value and turns them into a dictionary
        """
        parametersets = {}
        for arg in args:
            key = arg[0]
            parametersets[key] = arg[1].copy()
            self.parameter_names.append(key)
        return parametersets

    @staticmethod
    def make_parameter_combinations(_dictionary: dict[any]) -> list[dict[any]]:
        r"""
        description: 
        """
        keys = list(_dictionary.keys())
        _list = []
        list_with_dicts = []
        for key in keys:
            _list.append(_dictionary[key])
        combinations = list(itertools.product(*_list))
        for combination in combinations:
            _dict = {}
            for idx, entry in enumerate(combination):
                _dict[keys[idx]] = entry
            list_with_dicts.append(_dict.copy())
            _dict.clear()
        return list_with_dicts
    
    def run_experiments(self):
        r"""
        description: generate all possible combinations of experiment parameters
        for the experiment function,
        call the experiment function for every combination for every cross validation
        compute mean and std of the metrics over the cross validations
        """
        for exp_idx, combination in enumerate(self.combinations):
            print(f"Starting experiment number {exp_idx}")
            keys = list(combination.keys())  # get all parameter names
            _list = []
            for key in keys:
                _list.append(combination[key])  # get all parameter values
            cv_df = pandas.DataFrame(columns=self.metric_names)  # df for mean std cv calculation
            for cv_idx, data in enumerate(self.cross_val_data):
                metrics = self.exp_run_func(cv_idx, data, *_list)  # run an experiment with the given parameters and return the metrics
                temp_cv_df = pandas.DataFrame([metrics],columns=self.metric_names)
                cv_df = pandas.concat([cv_df, temp_cv_df], ignore_index=True)  # append metrics to df
            # compute mean and std values
            means = numpy.mean(cv_df, axis=0)
            means = list(means)
            stds = numpy.std(cv_df, axis=0)
            stds = list(stds)
            liste = []
            while means:
                liste.append(means.pop(0))
                liste.append(stds.pop(0))
            current_exp = [exp_idx+1]
            # mean_df = pandas.DataFrame([liste], columns=ext_names)
            # current_cross_val = [cv_idx+1]
            current_log = current_exp + _list + liste  # combine variables and metrics
            temp_df = pandas.DataFrame([current_log], columns=self.log_columns)  # save current log into df
            self.log_df = pandas.concat([self.log_df, temp_df], ignore_index=True)
            
            # append cv df mean std to exp df

    def experiment_logging(self, entries: list[any]):
        r"""
        description: log the variables and metrics for every experiment.
        The variables are defined by the experiment.
        The metrics are the result of the run_experiment function
        The log should be a pandas df
        """
        print("I'm not functional yet!")
        temp_df = pandas.DataFrame([entries],columns=self.log_columns)  # save current log into df
        self.log_df = pandas.concat([self.log_df, temp_df], ignore_index=True)
