from . import DEVICE, SPM_MODEL_FILE_PATH
import torch
from torch import nn

################################################k
### 翻訳機（ビーム探索）
################################################
from queue import PriorityQueue
from sentencepiece import SentencePieceProcessor

class BeamSearchNode(object):
  def __init__(self, decoder_input, log_prob, length):
    self.decoder_input = decoder_input
    self.log_prob = log_prob
    self.length = length

  def eval(self, alpha=0.6): # GNMTのペナルティ
    return self.log_prob / (((5 + self.length) / (5 + 1)) ** alpha)

  def __str__(self):
    return f'{self.decoder_input} length:{self.length}'

class Translator():
  def __init__(self, model):
    self.model = model.to(DEVICE)

  def tokenize_japanese(self, sentence: str):
    sp = SentencePieceProcessor(model_file=SPM_MODEL_FILE_PATH)
    tokenized_sentence = sp.encode(sentence)
    tensor = torch.tensor(tokenized_sentence)
    return tensor

  def translate(self, ja_sentence, max_sentence_length=50):
    tokenized_sentence = self.tokenize_japanese(ja_sentence).to(DEVICE)
    sp = SentencePieceProcessor(model_file=SPM_MODEL_FILE_PATH)

    dec = torch.tensor([1]).to(DEVICE)
    with torch.no_grad():
      for i in range(max_sentence_length):  # 最大生成トークン数
          mask = nn.Transformer.generate_square_subsequent_mask(dec.size(0)).to(DEVICE)
          output = self.model(tokenized_sentence, dec.unsqueeze(0), mask)

          # 最後のトークンの出力を取得
          next_word_logits = output[-1, -1, :]
          next_word = torch.argmax(next_word_logits, dim=-1)
          next_word = next_word.item()

          # 終了トークンが生成されたら停止
          if next_word == 2:  # 2は終了トークンのIDとする
              break

          # 生成されたトークンをdecに追加
          dec = torch.cat([dec, torch.tensor([next_word]).to(DEVICE)])

    return sp.decode(dec.tolist())


  def beam_translate(self, ja_sentence, max_sentence_length=50, beam_size=10):
    tokenized_sentence = self.tokenize_japanese(ja_sentence).to(DEVICE)
    initial_node = BeamSearchNode(torch.tensor([1]), 0, 1)
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
        mask = nn.Transformer.generate_square_subsequent_mask(best_node.decoder_input.size(0)).to(DEVICE)
        output = self.model(tokenized_sentence, best_node.decoder_input.unsqueeze(0).to(DEVICE), mask)

        next_word_logits = output[-1, -1, :]

        top_log_probs, top_indices = torch.topk(next_word_logits, beam_size)
        for beam in range(beam_size):
          predicted_id = top_indices.tolist()[beam]
          next_input = torch.tensor(best_node.decoder_input.tolist() + [predicted_id]).to(DEVICE)
          log_prob = top_log_probs.tolist()[beam]
          log_prob += best_node.log_prob
          node = BeamSearchNode(next_input, log_prob, best_node.length+1)
          priority_queue.put(( -node.eval(), node))

        queue_size += beam_size
      except Exception as e:
        print(e.args)
        pass


    sp = SentencePieceProcessor(model_file=SPM_MODEL_FILE_PATH) ##　日本語・英語の原文をちゃんとデコードできるか

    translated_sentence = sp.decode_ids(best_node.decoder_input.tolist())
    return translated_sentence

