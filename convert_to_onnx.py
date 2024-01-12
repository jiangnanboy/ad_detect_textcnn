import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

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
        print('x shape: {}'.format(x.shape[2]))
        x = [F.relu(conv(x)).squeeze(3) for conv in
             self.convs]  # len(filter_size) * (N, filter_num, H) -> 3 * (163, 100, 18)
        # MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False),stride默认为kernal_size

        x = [F.max_pool1d(output, (output.shape[2],)).squeeze(2) for output in
             x]  # len(filter_size) * (N, filter_num) -> 3 * (163, 100)

        x = torch.cat(x, 1)  # (N, filter_num * len(filter_size)) -> (163, 100 * 3)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def sentence_segment(sentence):
    sentence_words = list(sentence)
    if len(sentence_words) < 100:
        sent_range = 100 - len(sentence_words)
        for i in range(sent_range):
            sentence_words.append('<pad>')
    elif len(sentence_words) > 100:
        sentence_words = sentence_words[0:100]
    return sentence_words

def bow(sentence, words):
    sentence_words = sentence_segment(sentence)
    indexed = [words.stoi[t] for t in sentence_words]
    src_tensor = torch.LongTensor(indexed)
    src_tensor = src_tensor.unsqueeze(0)
    return src_tensor

def predict_class(sentence):
    sentence_bag = bow(sentence, word_dict)
    model.eval()
    with torch.no_grad():
        outputs = model(sentence_bag)
    print('outputs:{}'.format(outputs))
    predicted_prob, predicted_index = torch.max(F.softmax(outputs, 1), 1)  # 预测最大类别的概率与索引
    print('softmax_prob:{}'.format(predicted_prob))
    print('softmax_index:{}'.format(predicted_index))
    results = []
    # results.append({'intent':index_classes[predicted_index.detach().numpy()[0]], 'prob':predicted_prob.detach().numpy()[0]})
    results.append({'intent': predicted_index.detach().numpy()[0], 'prob': predicted_prob.detach().numpy()[0]})
    print('result:{}'.format(results))
    return results

def get_response(predict_result):
    tag = predict_result[0]['intent']
    return tag

def predict(text):
    predict_result = predict_class(text)
    res = get_response(predict_result)
    return res

words_path = os.path.join(os.getcwd(), "words.pkl")
with open(words_path, 'rb') as f_words:
    word_dict = pickle.load(f_words)

model = TextCNN(len(word_dict), 300, 16)
model_path = os.path.join(os.getcwd(), "model.h5")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def convert2onnx(sentence):
    sentence_bag = bow(sentence, word_dict)
    input_names = ['input_1']
    output_names = ['output_1']
    torch.onnx.export(
        model,
        sentence_bag,
         r'Z:\project\python\intent_classification-master\models\pred.onnx',
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        # dynamic_axes={
        #     input_names[0]: {1: 'input_1'}}
    )

if __name__ == '__main__':
    # print(predict("融合型企业建设培育试点的通知"))
    convert2onnx("关于开展产教融合型企业建设培育试点的通知关于开展产教融合型企业建设培育试点的通知")

