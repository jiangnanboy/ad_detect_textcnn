import os
import torch
from torchtext import data, datasets
from torchtext import data
from torchtext.vocab import Vectors
from torch import nn, optim
import torch.nn.functional as F
from torch import Tensor
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from typing import Optional,Sequence


DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

intent_classification_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

# 训练数据路径
train_data = os.path.join(intent_classification_path, 'ad_data/train.csv')
val_data = os.path.join(intent_classification_path, 'ad_data/test.csv')

# 读取数据
train_data = pd.read_csv(train_data, delimiter=',')
val_data = pd.read_csv(val_data, delimiter=',')

# 按字分
tokenize = lambda x: x.split(' ')

TEXT = data.Field(
    sequential=True,
    tokenize=tokenize,
    lower=True,
    use_vocab=True,
    pad_token='<pad>',
    unk_token='<unk>',
    batch_first=True,
    fix_length=500)

LABEL = data.Field(
    sequential=False,
    unk_token=None,
    use_vocab=True)


# 获取训练或测试数据集
def get_dataset(csv_data, text_field, label_field, test=False):
    fields = [('id', None), ('text', text_field), ('label', label_field)]
    examples = []
    if test:  # 测试集，不加载label
        for text in csv_data['text']:
            examples.append(data.Example.fromlist([None, text, None], fields))
    else:  # 训练集
        for text, label in zip(csv_data['text'], csv_data['label']):
            examples.append(data.Example.fromlist([None, str(text), label], fields))
    return examples, fields


train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
train = data.Dataset(train_examples, train_fields)

val_examples, val_fields = get_dataset(val_data, TEXT, LABEL)
val = data.Dataset(val_examples, val_fields)

# 预训练数据
# sogou的预训练向量可从这里下载(https://github.com/Embedding/Chinese-Word-Vectors
pretrained_embedding = os.path.join(os.getcwd(), 'sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5')
vectors = Vectors(name=pretrained_embedding)
print('vectors : {}'.format(vectors.dim))
# 构建词典
TEXT.build_vocab(train, val, min_freq=1, vectors=vectors)

words_path = os.path.join(os.getcwd(), 'ad_words.pkl')
with open(words_path, 'wb') as f_words:
    pickle.dump(TEXT.vocab, f_words)

LABEL.build_vocab(train, val, min_freq=1)
labels_path = os.path.join(os.getcwd(), 'ad_label.pkl')
with open(labels_path, 'wb') as f_labels:
    pickle.dump(LABEL.vocab, f_labels)

BATCH_SIZE = 16
# 构建迭代器
#train_dataset, val_dataset = train.split(split_ratio=0.9)
train_dataset, val_dataset = train, val

print('train_data size: {}'.format(len(train_dataset)))
print('val_data size: {}'.format(len(val_dataset)))

train_iter = data.BucketIterator(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    sort_within_batch=False)

val_iter = data.BucketIterator(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print('TEXT.vocab.vectors.shape: {}'.format(TEXT.vocab.vectors.shape))
print('label vocab : {}'.format(LABEL.vocab.itos))

# 构建分类模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, filter_num=100, filter_size=(3, 4, 5), dropout=0.5):
        '''
        vocab_size:词典大小
        embedding_dim:词维度大小
        output_size:输出类别数
        filter_num:卷积核数量
        filter_size(3,4,5):三种卷积核，size为3,4,5，每个卷积核有filter_num个，卷积核的宽度都是embedding_dim
        '''
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        # conv2d(in_channel,out_channel,kernel_size,stride,padding),stride默认为1，padding默认为0
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (k, embedding_dim)) for k in filter_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filter_num * len(filter_size), output_size)

    '''
    以下forward中的卷积和池化计算方式如下：

    1.卷积
    卷积后的shape公式计算简化为:np.floor((n + 2p - f)/s + 1)
    输入shape:(batch, in_channel, hin, win) = (163, 1, 20, 300)，20为句子长度，300为embedding大小
    输出shape:
    hout=(20 + 2 * 0 - 1 * (3 - 1) - 1)/1 + 1 = 18
    wout=(300 + 2 * 0 - 1 * (300 - 1) -1)/1 + 1 = 1
    =>
    output:(batch, out_channel, hout, wout) = (163, 100, 18, 1)

    2.max_pool1d池化
    简化公式：np.floor((l + 2p - f)/s + 1)
    输入shape:(N,C,L):(163, 100, 18, 1) -> squeeze(3) -> (163, 100, 18)
    输出shape:
    lout = (18 + 2*0 - 18)/18 +1 = 1 -> (163, 100, 1)
    '''

    def forward(self, x):
        # x :(batch, seq_len) = (163, 20)
        x = self.embedding(x)  # [batch,word_num,embedding_dim] = [N,H,W] -> (163, 20, 300)
        x = x.unsqueeze(1)  # [batch, channel, word_num, embedding_dim] = [N,C,H,W] -> (163, 1, 20, 300)
        x = [F.relu(conv(x)).squeeze(3) for conv in
             self.convs]  # len(filter_size) * (N, filter_num, H) -> 3 * (163, 100, 18)
        # MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False),stride默认为kernal_size
        x = [F.max_pool1d(output, output.shape[2]).squeeze(2) for output in
             x]  # len(filter_size) * (N, filter_num) -> 3 * (163, 100)
        x = torch.cat(x, 1)  # (N, filter_num * len(filter_size)) -> (163, 100 * 3)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def get_confusion_matrix(trues, preds):
    labels = [0, 1]
    conf_matrix = confusion_matrix(trues, preds, labels)
    return conf_matrix

def evaluate_accuracy(data_iter, net):
    val_preds = []
    val_trues = []
    acc_sum, n = 0.0, 0
    net.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            X, y = batch.text, batch.label
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            #X = X.permute(1, 0)
            #y.data.sub_(1)  #X转置 y下标从0开始
            if isinstance(net, torch.nn.Module):
                outputs = net(X)
                acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
                outputs = outputs.argmax(dim=1)
                val_preds.extend(outputs.detach().cpu().numpy())
                val_trues.extend(y.detach().cpu().numpy())
            else: # 自定义的模型
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    net.train()
    return acc_sum / n, val_trues, val_preds

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha:Optional[Tensor]=None,
                 gamma:float=0.,
                 reduction:str='mean',
                 ignore_index:int=-100):
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError('Reduction must be one of:"mean", "sum", "none".')
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x:Tensor, y:Tensor) -> Tensor:
        if x.ndim > 2:
            # (N,C,d1,d2,...,dk) --> (N*d1*...*dk, c)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N,d1,d2,...,dk) --> (N*d1*...*dk)
            y = y.view(-1)
        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        #compute weighted cross entropy term:-alpha*log(pt)
        #(alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term:(1-pt)^gamma
        pt = log_pt.exp()
        focal_term = (1-pt) ** self.gamma

        # the full loss:-alpha*((1-pt)^gamma)*log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def focal_loss(alpha:Optional[Sequence]=None,
               gamma:float=2.,
               reduction:str='mean',
               ignore_index:int=-100,
               device='gpu:1',
               dtype=torch.float32) -> FocalLoss:
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=DEVICE, dtype=dtype)

        f1 = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction, ignore_index=ignore_index)

    return f1

# 构建model
model = TextCNN(len(TEXT.vocab), TEXT.vocab.vectors.shape[1], 2).to(DEVICE)
# 利用预训练模型初始化embedding，requires_grad=True，可以fine-tune
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
# 训练模式
model.train()
# 优化和损失
#optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, weight_decay=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()
#criterion = focal_loss([0.1, 0.1, 0.125, 0.125, 0.125, 0.125, 0.15, 0.15])

acc = 0
for iter in range(1000):
    batch_count = 0
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for i, batch in enumerate(train_iter):
        train_text = batch.text
        train_label = batch.label
        train_text = train_text.to(DEVICE)
        train_label = train_label.to(DEVICE)
        out = model(train_text)
        loss = criterion(out, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_l_sum += loss.item()
        train_acc_sum += (out.argmax(dim=1) == train_label).sum().item()
        n += train_label.shape[0]
        batch_count += 1

    test_acc, val_trues, val_preds = evaluate_accuracy(val_iter, model)
    print(
        'epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
        % (iter + 1, train_l_sum / batch_count, train_acc_sum / n,
           test_acc))
    conf_matrix = get_confusion_matrix(val_trues, val_preds)
    print(conf_matrix)
    print(classification_report(val_trues, val_preds))

    if test_acc >= acc:
        acc = test_acc
        model_path = os.path.join(os.getcwd(), "ad_model.h5")
        torch.save(model.state_dict(), model_path)





