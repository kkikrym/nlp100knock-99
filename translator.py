import torch
from torch import nn
import MeCab
mecab = MeCab.Tagger()
mecab.parse("")

JA_MODEL_FILE_PATH = 'ja_sentencepiece_model.model'
EN_MODEL_FILE_PATH = 'en_sentencepiece_model.model'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

################################################k
### エンコーダ・デコーダ（LSTM）
### 
### Dropout層を追加、中間層を3層に増加
################################################

class Encoder(nn.Module):
  def __init__(self, vocab_size, emb_size, padding_idx, hidden_size, ):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx)
    self.dropout = nn.Dropout(0.5)
    self.model = nn.LSTM(emb_size, hidden_size, num_layers=3, batch_first=True)

  def forward(self, input, h0=None):
    x = self.embedding(input)
    x = self.dropout(x)
    x, state = self.model(x, h0)
    return x, state

class Decoder(nn.Module):
  def __init__(self, vocab_size, emb_size, padding_idx, hidden_size, output_size):
    super(Decoder, self).__init__()
    self.padding_idx = padding_idx

    self.embedding = nn.Embedding(vocab_size, emb_size,)
    self.dropout = nn.Dropout(0.5)

    self.model = nn.LSTM(emb_size, hidden_size, num_layers=3, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size,)

  def forward(self, input, hidden, h0=None):
    x = self.embedding(input)
    x = self.dropout(x)
    x, state = self.model(x, hidden)
    x = self.fc(x)
    return x, state

################################################k
### 翻訳機（ビーム探索）
################################################
from queue import PriorityQueue
class BeamSearchNode(object):
  def __init__(self, previous_node, decoder_input, decoder_state, log_prob, length):
    self.previous_node = previous_node
    self.decoder_input = decoder_input
    self.decoder_state = decoder_state
    self.log_prob = log_prob
    self.length = length

  def eval(self, alpha=0.6): # GNMTのペナルティ
    return self.log_prob / (((5 + self.length) / (5 + 1)) ** alpha)

  def __str__(self):
    return f'{self.decoder_input} length:{self.length}'

from sentencepiece import SentencePieceProcessor
class Translator():
  def __init__(self, encoder, decoder):
    self.encoder = encoder.to('cpu')
    self.decoder = decoder.to('cpu')

  def tokenize_japanese_by_mecab(self, sentence: str):
    data = mecab.parse(sentence)
    surfaces = []
    words = data.split('\n')

    for word in words:
      if word.startswith('EOS'): # 文の終わりを回避
        continue

      word = word.split('\t')
      if word[0] == '\u3000': # 空白を回避
        continue
      if word[0] == '': # 空白を回避
        continue

      if word[4].split('-')[0].strip() == '記号':
        continue

      surface = word[0]
      surfaces.append(surface)

    return surfaces

  def tokenize_japanese(self, sentence: str):

    # Mecabで形態素に分解したものをトークンにする
    sentence = self.tokenize_japanese_by_mecab(sentence)

    sp = SentencePieceProcessor(model_file=JA_MODEL_FILE_PATH)
    tokenized_sentence = []
    for word in sentence[0]:
      tokenized_sentence += sp.encode(word)
    tensor = torch.tensor(tokenized_sentence)
    return tensor

  def beam_translate(self, ja_sentence:str, beam_size=10, max_sentence_length=50):
    with torch.no_grad():
      tokenized_sentence = self.tokenize_japanese(ja_sentence)
      _, state = self.encoder(tokenized_sentence)

    initial_node = BeamSearchNode(None, torch.tensor([1]), state, 0, 1)
    selected_node = None

    priority_queue = PriorityQueue() # スコアが小さいものから順に取り出される
    priority_queue.put((-initial_node.eval(), initial_node))
    queue_size = 1
    while True:
      try:
        score, best_node = priority_queue.get()

        if queue_size > 2000:
          print('max queue size exeeded')
          selected_node = best_node
          break

        if best_node.length > max_sentence_length:
          selected_node = best_node
          break

        decoder_output, state = self.decoder(best_node.decoder_input, state)
        top_log_probs, top_indices = torch.topk(decoder_output, 10)
        for beam in range(beam_size):
          predicted_id = top_indices.tolist()[0][beam]
          next_input = torch.tensor([predicted_id])
          log_prob = top_log_probs.tolist()[0][beam]
          log_prob += best_node.log_prob
          node = BeamSearchNode(best_node, next_input, state, log_prob, best_node.length+1)
          priority_queue.put(( -node.eval(), node))

        queue_size += beam_size
      except Exception as e:
        print(e.args)
        pass

    predicted_ids = []
    sp = SentencePieceProcessor(model_file=EN_MODEL_FILE_PATH)

    while selected_node.previous_node:
      predected_id = int(selected_node.decoder_input)
      selected_node = selected_node.previous_node
      predicted_ids.insert(0, predected_id)

    translated_sentence = sp.decode_ids(predicted_ids)

    return translated_sentence