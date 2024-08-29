import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLayer, BertEmbeddings, BertPooler


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.uniform_(self.weight, -0.1, 0.1)  # 权重初始化

    def forward(self, x):
        # 计算注意力得分
        scores = torch.matmul(x, self.weight)
        attention_weights = torch.softmax(scores, dim=1)

        # 应用注意力权重并求和
        weighted_output = x * attention_weights.unsqueeze(-1).expand_as(x)
        output = weighted_output.sum(1)
        return output


class TextCNN(nn.Module):
    def __init__(self, input_channels, num_filters, filter_sizes):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_channels, out_channels=num_filters, kernel_size=f_size, padding=f_size // 2)
            for f_size in filter_sizes
        ])

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch_size, channels, sequence_length]
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2) for conv_output in x]
        x = torch.cat(x, 1)
        return x

class TextClassifier(nn.Module):
    def __init__(self, num_labels=2, num_filters=100, filter_sizes=[3, 4, 5]):
        super(TextClassifier, self).__init__()
        model_path = '/home/liaoguoan/workdir/JointMatch-main/code/pre/bert-base-uncased'
        self.bert = BertModel.from_pretrained(model_path)
        self.attention = Attention(hidden_size=768)  # 假设隐藏层大小为768

        # TextCNN部分
        self.text_cnn = TextCNN(input_channels=768, num_filters=num_filters, filter_sizes=filter_sizes)

        # 由于我们添加了TextCNN层，线性层的输入需要调整
        self.linear = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes), 128),  # 注意这里的变化
            nn.Tanh(),
            nn.Linear(128, num_labels)
        )

    def forward(self, inputs):
        # 从输入中提取特征
        features = self.extract_features(inputs)

        # 用提取的特征进行分类
        logits = self.linear(features)
        return logits

    def extract_features(self, inputs):
        outputs = self.bert(**inputs)
        sequence_output = outputs.last_hidden_state

        #print("Shape after BERT:", sequence_output.shape)  # 检查形状

        attention_output = self.attention(sequence_output)
        # 在BERT输出上应用注意力机制
        attention_output = attention_output.unsqueeze(1)  # 形状变为 [batch_size, 1, hidden_size]
        #print("Shape after Attention:", attention_output.shape)  # 检查形状

        # 从TextCNN层提取特征
        cnn_output = self.text_cnn(attention_output)
        #print("Shape after TextCNN:", cnn_output.shape)  # 检查形状

        return cnn_output
