import predict
import nltk
import re
nltk.download('words')
from nltk.corpus import words
import difflib


def answer(imagepath):
    sp = '[@_!#$%^&*()<>?.,/\|}{~:]'
    answer = predict.predict(imagepath)
    print(answer)
    word_list = words.words()
    arr = difflib.get_close_matches(answer[1:], word_list)
    if(len(arr) > 0):
        return arr[0]
    else:
        if(answer in sp or answer[1:] in sp):
            return ""
        return answer[1:]
