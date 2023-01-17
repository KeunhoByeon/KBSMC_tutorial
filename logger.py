import numpy as np


class Logger:
    def __init__(self, log_path: str = 'log.txt', epochs: int = 0, dataset_size: int = 0, components: list = [], float_round: int = -1):
        self.log_path = log_path
        self.epochs = epochs
        self.dataset_size = dataset_size

        self.set_float_round(float_round)
        self.set_sort(components)

        self.current_line = ''
        self.current_components = []
        self.wf = open(self.log_path, 'w')

        self.history = {}

    def __del__(self):
        self.wf.close()

    def add_history(self, history_key, data: dict):
        if history_key not in self.history.keys():
            self.history[history_key] = {}

        for key, item in data.items():
            if key not in self.history[history_key].keys():
                self.history[history_key][key] = []
            self.history[history_key][key].append(item)

    def get_history_data(self, history_key, data_key):
        return np.mean(self.history[history_key][data_key])

    def set_sort(self, components: list):
        self.components = components

    def set_float_round(self, float_round: int):
        self.float_round = float_round

    def _add_component(self, key, data):
        if isinstance(data, float) and self.float_round > 0:
            data = round(data, self.float_round) if round(data, self.float_round) != 0. else data
            data = str(data)
            if len(data.split('.')) > 1 and len(data.split('.')[1]) < self.float_round:
                for _ in range(self.float_round - len(data.split('.')[1])):
                    data += '0'
        self.current_components.append('{}: {}'.format(key, data))

    def _make_line(self, titles, comp_dict):
        self.current_components = []
        titles = list(titles)

        title = ''
        if 'epoch' in comp_dict.keys():
            titles = ['[{}/{}]'.format(comp_dict['epoch'], self.epochs)] + titles
        if len(titles) > 0:
            title += '{}'.format(' '.join(titles))
        if 'batch' in comp_dict.keys():
            title += '[{}/{}]'.format(comp_dict['batch'], self.dataset_size)
        if len(title) > 0:
            self.current_components.append(title)

        if 'history_key' in comp_dict.keys():
            history_key = comp_dict['history_key']
            if history_key in self.history.keys():
                for key in self.components:
                    if key in self.history[history_key].keys():
                        self._add_component(key, self.get_history_data(history_key, key))
                        del (self.history[history_key][key])
                for key in sorted(self.history[history_key].keys()):
                    if key not in self.components and key not in ['epoch', 'batch', 'history_key']:
                        self._add_component(key, self.get_history_data(history_key, key))
                        del (self.history[history_key][key])

        for key in self.components:
            if key in comp_dict.keys():
                self._add_component(key, comp_dict[key])

        for key in sorted(comp_dict.keys()):
            if key not in self.components and key not in ['epoch', 'batch', 'history_key']:
                self._add_component(key, comp_dict[key])

        self.current_line = '  '.join(self.current_components)

    def write_log(self, *titles, **kwargs):
        self._make_line(titles, kwargs)
        self.wf.write(self.current_line + '\n')

    def print_log(self, *titles, **kwargs):
        self._make_line(titles, kwargs)
        print(self.current_line)

    def print_and_write_log(self, *titles, **kwargs):
        self._make_line(titles, kwargs)
        self.wf.write(self.current_line + '\n')
        print(self.current_line)

    def __call__(self, *titles, **kwargs):
        self._make_line(titles, kwargs)
        self.wf.write(self.current_line + '\n')
        print(self.current_line)