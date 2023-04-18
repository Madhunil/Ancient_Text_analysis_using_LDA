# imports
import sys
from collections import Counter
import numpy as np
import random


"""
A Language Model to generate random words after training on various ancient Indo-European classics.

This LM class will be used to produce numerous sentences using various ancient texts and different n-grams to see patterns in sentence generation after training on various Indo-European languages.

@author: Madhunil Pachghare
"""


# helper functions

def make_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
    Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

    Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
    """
    n_gram_list = []
    for i in range(len(tokens)):
        if len(tokens[i:i+n]) > n-1:
            n_gram_list.append(tokens[i:i+n])
    return n_gram_list
    

class LanguageModel:
  # constants to define pseudo-word tokens
  UNK = "<UNK>"
  SENT_BEGIN = "<s>"
  SENT_END = "</s>"

  def __init__(self, n_gram, is_laplace_smoothing):
    """Initializes an untrained LanguageModel
    Parameters:
      n_gram (int): the n-gram order of the language model to create
      is_laplace_smoothing (bool): whether or not to use Laplace smoothing
    """
    self.n_gram = int(n_gram)
    self.is_laplace_smoothing = bool(is_laplace_smoothing)
    pass


  def train(self, training_file_path):
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Parameters:
      training_file_path (str): the location of the training data to read

    Returns:
    None
    """
    self.training_file_path = str(training_file_path)
    
    berp_training = open(training_file_path, "r", encoding='utf-8')
    contents_berp_train = berp_training.read()
    berp_training.close()
    
    contents_berp_train_tokens = contents_berp_train.split()
    self.token_count = len(contents_berp_train_tokens)
    
    berp_train_tokens_count = Counter(contents_berp_train_tokens)
    
    
    # Converting words with frequency 1 to <UNK>
    unk_list = []
    for i in berp_train_tokens_count.items():
        if i[1] == 1:
            unk_list.append(i[0])
    for x in range(len(contents_berp_train_tokens)):
        if contents_berp_train_tokens[x] in unk_list:
            contents_berp_train_tokens[x] = '<UNK>'
    
    
    self.vocabulary_berp = set(contents_berp_train_tokens)
    
    # Getting count for n-1 grams i.e. denominators of the score function
    if self.n_gram > 1:
        n_gram_list_prev = make_ngrams(contents_berp_train_tokens,self.n_gram - 1)
        n_gram_prev_count = Counter(tuple(y) for y in n_gram_list_prev)
        self.n_gram_prev_count = dict(n_gram_prev_count)
    else:
        n_gram_prev_count = Counter(contents_berp_train_tokens)
        self.n_gram_prev_count = dict(n_gram_prev_count)
    
    n_gram_list = make_ngrams(contents_berp_train_tokens,self.n_gram)
    
    n_gram_list_count = Counter(tuple(y) for y in n_gram_list)
    
    
    # Getting count for each n-gram i.e. numerators of the score function
    self.n_gram_count_dict = dict(n_gram_list_count)
    
    pass

  def score(self, sentence):
    """Calculates the probability score for a given string representing a single sentence.
    Parameters:
      sentence (str): a sentence with tokens separated by whitespace to calculate the score of
      
    Returns:
      float: the probability value of the given string for this model
    """
    test_data_sentence_tokens = sentence.split()
    
    # Converting unknown words to <UNK>
    for i in range(len(test_data_sentence_tokens)):
        if test_data_sentence_tokens[i] not in self.vocabulary_berp:
            test_data_sentence_tokens[i] = '<UNK>'
    
    n_gram_test = make_ngrams(test_data_sentence_tokens,self.n_gram)
    
    
    if self.is_laplace_smoothing:
        probability = []
        for wi_w in n_gram_test:
            wi_w = tuple(wi_w)
            
            # Checking if the count is zero
            
            if wi_w not in self.n_gram_count_dict:
                count_up = 0 + 1
            else:
                count_up = self.n_gram_count_dict[wi_w] + 1
            
            
            
            
            if self.n_gram > 1:
                count_down = self.n_gram_prev_count[wi_w[0:self.n_gram-1]] + len(self.vocabulary_berp)
            else:
               
               count_down = self.token_count + len(self.vocabulary_berp)
               
            
            
            probability.append(count_up/count_down)
        # Multiplying all probabilities
        score = np.prod(probability)
    else:
        probability = []
        for wi_w in n_gram_test:
            wi_w = tuple(wi_w)
            if wi_w not in self.n_gram_count_dict:
                count_up = 0
            else:
                count_up = self.n_gram_count_dict[wi_w]
            
            
            
            if self.n_gram > 1:
                count_down = self.n_gram_prev_count[wi_w[0:self.n_gram-1]]
            else:
               
               count_down = self.token_count
            
            probability.append(count_up/count_down)
        # Multiplying all probabilities
        score = np.prod(probability)
    
    return score
    
    pass

  def generate_sentence(self):
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      str: the generated sentence
    """
    # Initializing sentence for different n-grams
    if self.n_gram > 1:
        sentence = (self.n_gram-1)*['<s>']
    else:
        sentence = ['<s>']
    
    check_word = tuple((self.n_gram)*['<s>'])
    
    # Running the loop till </s> is found
    while(True):
        next_word_dict = {}
        for key,value in self.n_gram_count_dict.items():
            if key[0:self.n_gram-1] == check_word[1:self.n_gram]:
                next_word_dict[key] = value
        
        next_word = []
        probability_gen_n = []
        for word in next_word_dict:
            next_word.append(word)
            word = tuple(word)
            count_up = next_word_dict[word]
            if self.n_gram > 1:
                count_down = self.n_gram_prev_count[word[0:self.n_gram-1]]
            else:
               count_down = self.token_count
             # Getting probabilities for all selected tokens  
            probability_gen_n.append(count_up/count_down)
        next_word_str = list(map(' '.join, next_word))
        
        # Sampling a token from given choices based on probability distribution
        generated_word_choices = random.choices(next_word_str,weights=probability_gen_n,k=1)
        
        put_in = str(*generated_word_choices).split()
        check_word = tuple(str(*generated_word_choices).split())
        
        if check_word[self.n_gram-1] == '<s>':
            pass
        elif check_word[self.n_gram-1] == '</s>':
            if self.n_gram > 1:
                t = (self.n_gram-1)*['</s>']
                
                sentence = sentence+(t)
            else:
                sentence.append('</s>')
            
            break
        else:
            sentence.append(str(put_in[self.n_gram-1]))
        pass
    # Combining all tokens to form a final sentence
    final_sentence = ' '.join(sentence)
    return final_sentence


  def generate(self, n):
    """Generates n sentences from a trained language model using the Shannon technique.
    Parameters:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing strings, one per generated sentence
    """
    final_sentence_list = []
    for i in range(n):
        final_sentence = self.generate_sentence()
        final_sentence_list.append(final_sentence)
    return final_sentence_list
    pass
  
  def perplexity(self, test_sequence):

    """Measures the perplexity for the given test sequence with this trained model. 

             As described in the text, you may assume that this sequence 

             may consist of many sentences "glued together".

    Parameters:

      test_sequence (string): a sequence of space-separated tokens to measure the perplexity of

    Returns:

      float: the perplexity of the given sequence

    """
    test_sequence_score = self.score(test_sequence)
    test_sequence_tokens = test_sequence.split()
    n = len(test_sequence_tokens)
    # Handling edge case of probability score being zero
    if test_sequence_score!=0:
        prob_inverse = 1/test_sequence_score
        perplexity = prob_inverse**(1/n)
        return perplexity
    else:
        return float('inf')
    
    
#%%

def main():
  # TODO: implement
  # Getting training and testing paths from system input
  training_path = sys.argv[1]
  
  # Incase we have a test set
  testing_path = sys.argv[2]
  
  
  # Creating object for unigram model, similar strategy will be used for other n-grams.
  LM_unigram_with_laplace = LanguageModel(1,True)
  print('Model: unigram, laplace smoothed')
  # Training the unigram model
  LM_unigram_with_laplace.train(training_path)
  # Generating 50 random sentences using unigram model
  print('Sentences:')
  sentence_list = LM_unigram_with_laplace.generate(50)
  for i in sentence_list:
    print(i)
  
  score_list = []
  
  
  # Incase we have a test set
  for sentence in test_data_sentences:
      score_list.append(LM_unigram_with_laplace.score(sentence))
    
  # Getting Average probability
  score_unigram_with_laplace = sum(score_list)/len(score_list)
  
  # Getting first ten sentences of the test corpus
  first_10 = []
  for i in range(10):
      first_10.append(test_data_sentences[i])
  first_10_list = ' '.join(first_10)
  
  perplexity_unigram_with_laplace = LM_unigram_with_laplace.perplexity(first_10_list)
  
  print("# of test sentences:",len(score_list))
  print('Average probability:',score_unigram_with_laplace)
  print('Standard deviation:',np.std(score_list))
  print('Perplexity for 1-grams:',perplexity_unigram_with_laplace)
  
  
  # Another n-gram example below, similar strategy will be used for higher order n-grams.
  
  # Creating object for bigram model
  LM_bigram_with_laplace = LanguageModel(2,True)
  print('        ')
  print('Model: bigram, laplace smoothed')
  # Training the bigram model
  LM_bigram_with_laplace.train(training_path)
  # Generating 50 random sentences using unigram model
  print('Sentences:')
  sentence_list = LM_bigram_with_laplace.generate(50)
  for i in sentence_list:
      print(i)
  
  score_list = []
  
  for sentence in test_data_sentences:
      score_list.append(LM_bigram_with_laplace.score(sentence))
  # Getting Average probability
  score_bigram_with_laplace = sum(score_list)/len(score_list)
  
  # Getting first ten sentences of the test corpus
  first_10 = []
  for i in range(10):
      first_10.append(test_data_sentences[i])
  first_10_list = ' '.join(first_10)
  
  perplexity_bigram_with_laplace = LM_bigram_with_laplace.perplexity(first_10_list)
  
  print("# of test sentences:",len(score_list))
  print('Average probability:',score_bigram_with_laplace)
  print('Standard deviation:',np.std(score_list))
  print('Perplexity for 2-grams:',perplexity_bigram_with_laplace)
  
  
  
  pass

    
if __name__ == '__main__':
    
  # make sure that they've passed the correct number of command line arguments
  if len(sys.argv) != 2 or len(sys.argv) != 3:
    print("Usage:", "python lm.py training_file.txt testingfile.txt")
    sys.exit(1)

  main()