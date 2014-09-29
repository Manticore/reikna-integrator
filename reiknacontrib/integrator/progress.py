from progressbar import Widget, WidgetHFill


class StatefulLabel(Widget):

    def __init__(self, display=None, time_formatter='.3f'):
        keys = []
        formatters = []
        if display is not None:
            for key in display:
                if isinstance(key, tuple):
                    key, formatter = key
                    formatter = "{" + key + ":" + formatter + "}"
                else:
                    formatter = "{" + key + "}"
                keys.append(key)
                formatters.append(formatter)

        format_str = ", ".join(key + ": " + formatter for key, formatter in zip(keys, formatters))

        self.keys = keys
        self.format_str = format_str
        self.time_format_str = (
            "time: {time:" + time_formatter + "}" + (", " if display is not None else ""))
        self.values = None

    def set(self, t, sample_dict):
        self.time = t
        if self.values is None:
            self.values = {}
        for key in self.keys:
            self.values[key] = sample_dict[key]['mean']

    def update(self, pbar):
        if self.values is not None:
            return (
                self.time_format_str.format(time=self.time) +
                self.format_str.format(**self.values))
        else:
            return "(not initialized)"


class HFill(WidgetHFill):

    def update(self, hbar, width):
        return ' ' * width
