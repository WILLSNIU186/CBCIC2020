import matplotlib.pyplot as plt

class Preprocessing():

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def apply_filter(self,data, hi, low, order):
        self.hi = hi
        self.low = low
        iir_params = dict(order=order, ftype='butter')
        self.filtered_data = data.copy().filter(l_freq=low, h_freq=hi, method='iir', iir_params=iir_params)
        fig = self.filtered_data.plot(scalings='auto')
        fig.canvas.key_press_event('a')
        fig.subplots_adjust(top=0.9)
        fig.suptitle('bdf butterworth order = {}'.format(order))
        plt.show()
        return self.filtered_data

    def create_evoked_data(self, data, selected_samples):
        self.evoked_data = data[selected_samples[0]].average()
        # self.evoked_data.plot()
        return self.evoked_data
