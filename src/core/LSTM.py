import torch 
import torch.nn as nn



class LSTM(nn.Module):
    def __init__(self, hidden_size=128, sequence_length=7, device="cuda"):
        super().__init__()
        self.hidden = hidden_size
        self.seq_len = sequence_length
        self.device=device
        # expected input shape B * 7 * 1
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden, num_layers=1, batch_first=True)
        self.linear = nn.Linear(self.hidden, 1)

    def forward(self, x):
        #x is of shape B * 7 * 1
        batch_size = x.size(0)

        h_t = torch.zeros(1, batch_size, self.hidden, device=self.device)
        c_t = torch.zeros(1, batch_size, self.hidden, device=self.device)

        outputs = []
        x_t = torch.rand(batch_size, 1, 1, device=x.device)
        for i in range(self.seq_len):
            out, (h_t, c_t) = self.lstm(x_t, (h_t, c_t))
            pred = self.linear(out)
            outputs.append(pred)
            x_t = x [:,i,:].unsqueeze(1)

        return torch.cat(outputs, dim=1)

    def sample(self):

        batch_size = 1
        h_t = torch.zeros(1, batch_size, self.hidden, device=self.device)
        c_t = torch.zeros(1, batch_size, self.hidden, device=self.device)

        outputs = []
        x_t = torch.rand(batch_size, 1, 1, device=self.device)
        with torch.no_grad():
            for i in range(self.seq_len):
                out, (h_t, c_t) = self.lstm(x_t, (h_t, c_t))
                pred = self.linear(out)
                outputs.append(pred)
                x_t = pred

        return torch.cat(outputs, dim=2).squeeze((0, 1))




