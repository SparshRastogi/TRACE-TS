from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super(ConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(
            self.input_filters,
            self.nb_filters,
            (self.filter_width, 1),
            dilation=(self.dilation, 1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            self.nb_filters,
            self.nb_filters,
            (self.filter_width, 1),
            dilation=(self.dilation, 1),
        )
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)
        return out


class DeepConvLSTM(nn.Module):
    def __init__(self, config):
        super(DeepConvLSTM, self).__init__()
        self.window_size = config["window_size"]
        self.nb_channels = config["nb_channels"]
        self.nb_classes = config["nb_classes"]
        self.nb_conv_blocks = config["nb_conv_blocks"]
        self.nb_filters = config["nb_filters"]
        self.filter_width = config["filter_width"]
        self.dilation = config["dilation"]
        self.batch_norm = config["batch_norm"]
        self.nb_units_lstm = config["nb_units_lstm"]
        self.nb_layers_lstm = config["nb_layers_lstm"]
        self.drop_prob = config["drop_prob"]
        self.weights_init = config["weights_init"]
        self.seed = config["seed"]
        self.conv_blocks = nn.ModuleList()
        for i in range(self.nb_conv_blocks):
            input_filters = 1 if i == 0 else self.nb_filters
            self.conv_blocks.append(
                ConvBlock(
                    self.filter_width,
                    input_filters,
                    self.nb_filters,
                    self.dilation,
                    self.batch_norm,
                )
            )
        self.final_seq_len = self.window_size - (self.filter_width - 1) * (
            self.nb_conv_blocks * 2
        )
        self.lstm_layers = nn.ModuleList()
        for i in range(self.nb_layers_lstm):
            input_size = (
                self.nb_channels * self.nb_filters if i == 0 else self.nb_units_lstm
            )
            self.lstm_layers.append(
                nn.LSTM(input_size, self.nb_units_lstm, batch_first=True)
            )
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.window_size, self.nb_channels)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, self.final_seq_len, self.nb_filters * self.nb_channels)
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def number_of_parameters(self):
        return sum((p.numel() for p in self.parameters() if p.requires_grad))


def init_weights(network):
    for m in network.modules():
        if isinstance(m, ConvBlock):
            _init_param(m.conv1.weight, network.weights_init)
            _init_param(m.conv2.weight, network.weights_init)
            if m.conv1.bias is not None:
                m.conv1.bias.data.fill_(0.0)
            if m.conv2.bias is not None:
                m.conv2.bias.data.fill_(0.0)
        elif isinstance(m, nn.Linear):
            _init_param(m.weight, network.weights_init)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    _init_param(param.data, network.weights_init)
                elif "bias" in name:
                    param.data.fill_(0.0)


def _init_param(tensor, method):
    if method == "xavier_normal":
        nn.init.xavier_normal_(tensor)
    elif method == "orthogonal":
        nn.init.orthogonal_(tensor)
    elif method == "kaiming_normal":
        nn.init.kaiming_normal_(tensor)
    elif method == "normal":
        nn.init.normal_(tensor)
    elif method == "xavier_uniform":
        nn.init.xavier_uniform_(tensor)
    elif method == "kaiming_uniform":
        nn.init.kaiming_uniform_(tensor)
