from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset


class TorchZarrDataloader(IterableDataset):
    def __init__(self, dataset, input_shape, output_shape, **kwargs):
        self.dataset = dataset
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __iter__(self):
        return iter(self.dataset.request_batch(self.input_shape, self.output_shape))
