import  torch
from torch.nn.utils.rnn import pad_sequence
import time
from datetime import timedelta

class Config():
    def __init__(self):
        self.model_name = 'BiLSTM_CNN'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_path = './saved_dict/best_model.pth'
        self.log_path = './log'

        self.require_improvement = 1000
        self.num_epoches = 20
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.class_list = ['Human', 'AI']
        self.num_classes = len(self.class_list)

        self.embed_dim = 3072
        # LSTM parameters
        self.dropout_LSTM = 0
        self.hidden_size = 256
        self.num_layers = 1
        # CNN parameters
        self.dropout_CNN = 0.5
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256

        self.dropout = 0.3

def get_time_diff(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
def collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_features, labels