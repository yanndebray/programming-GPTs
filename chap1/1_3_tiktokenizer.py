import tiktoken

encoding = tiktoken.encoding_for_model('gpt-4')
# encoding = tiktoken.get_encoding('cl100k_base')

def tokenize(text):
    return encoding.encode(text)

def detokenize(tokens):
    return encoding.decode(tokens)

if __name__ == '__main__':
    text = 'A long time ago in a galaxy far, far away...'
    tokens = tokenize(text)
    print(tokens) # [32, 1317, 892, 4227, 304, 264, 34261, 3117, 11, 3117, 3201, 1131]
    print(detokenize(tokens))