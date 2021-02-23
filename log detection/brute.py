from itertools import product
import pandas as pd

data = pd.Series('0')

chars = '0123456789QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm' # Chars Dictionary 

for length in range(1, 5): # length 1 to 4 
    to_attempt = product(chars, repeat=length) 
    attempt = pd.Series("".join(i) for i in to_attempt)
    data=attempt.append(attempt)
data.to_csv('log detection/dataset/brute.csv')
