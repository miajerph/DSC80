# lab.py


import pandas as pd
import numpy as np
import os
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------

def match_1(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    pattern = r'^..[\[].{2}[\]]'
     # ^.. matches any two characters at the start
    # \[ matches the '[' character literally
    # .{2} matches any two characters
    # \] matches the ']' character literally
    
    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None

def match_2(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    pattern = r'^\(858\)\s\d{3}-\d{4}$'
    # ^\(858\) matches '(858)' at the start
    # \s matches the space
    # \d{3}-\d{4} matches three digits, a hyphen, and four digits
    
    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None

def match_3(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    pattern = r'^[\w\s?]{5,9}\?$'
     # ^[\w\s?]{5,9} matches 5 to 9 alphanumeric, whitespace, or '?' characters at the start
    # \?$ matches a '?' character at the end

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None

def match_4(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False
    >>> match_4("$iiuABc")
    False
    >>> match_4("123$$$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    >>> match_4("$!@$")
    False
    """
    pattern = r'^\$[^abc$]*\$(?=[aA]+[bB]+[cC]+$)[aAbBcC]*$'
     # ^\$ matches a '$' at the start
    # [^abc$]* matches any character except 'a', 'b', 'c', or '$' zero or more times
    # \$ matches a '$' character literally
    # (?=[aA]+[bB]+[cC]+$) is a lookahead to ensure the rest of the string has 'a's followed by 'b's followed by 'c's
    # [aAbBcC]*$ matches any combination of 'a', 'A', 'b', 'B', 'c', or 'C' zero or more times at the end

    
    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None

def match_5(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """
    pattern = r'^[\w]+\.(py|pyw)$'
     # ^[\w]+ matches one or more word characters (letters, digits, underscores)
    # \. matches a literal dot
    # (py|pyw)$ matches 'py' or 'pyw' at the end of the string
    
    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None

def match_6(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """
    pattern = r'^[a-z]+_[a-z]+$'
    # ^[a-z]+ matches one or more lowercase letters at the start
    # _ matches a literal underscore
    # [a-z]+$ matches one or more lowercase letters at the end
    
    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None

def match_7(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """
    pattern = r'^_.+_$'
     # ^_ matches an underscore at the start
    # .+ matches one or more of any character (except newlines)
    # _$ matches an underscore at the end
    
    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None

def match_8(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo!!!!!!! !!!!!!!!!")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """
    pattern = r'^[^Oi1]+$'
    # ^[^Oi1]+$ matches one or more characters that are not 'O', 'i', or '1'

     # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None

def match_9(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    >>> match_9('NY-32-LAX-0000') # If the state is NY, the city can be any 3 letter code, including LAX or SAN!
    True
    >>> match_9('TX-32-SAN-4491')
    False
    '''
    pattern = r'^(NY-\d{2}-[A-Z]{3}-\d{4}|CA-\d{2}-(SAN|LAX)-\d{4})$'
    # ^(NY-\d{2}-[A-Z]{3}-\d{4}|[A-Z]{2}-\d{2}-(SAN|LAX)-\d{4})$ matches:
    # - 'NY' followed by a hyphen, two digits, another hyphen, three uppercase letters, another hyphen, and four digits
    # - OR a two-letter uppercase state code, a hyphen, two digits, a hyphen, 'SAN' or 'LAX', a hyphen, and four digits
    
    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None

def match_10(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_10('ABCdef')
    ['bcd']
    >>> match_10(' DEFaabc !g ')
    ['def', 'bcg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10('Ab..DEF')
    ['bde']
    
    '''
    # Convert the string to lowercase
    string = string.lower()
    # Remove all non-alphanumeric characters and the letter 'a'
    string = re.sub(r'[^0-9b-z]', '', string)
    # Extract every non-overlapping three-character substring
    return [string[i:i+3] for i in range(0, len(string), 3) if len(string[i:i+3]) == 3]

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

def extract_personal(s):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    bitcoin_pattern = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|\bbc1[a-zA-HJ-NP-Z0-9]{11,71}\b'
    street_pattern = r'\b\d+\s[A-Za-z0-9\s.,]+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Way|Ct|Court)\b'
    
    emails = re.findall(email_pattern, s)
    ssns = re.findall(ssn_pattern, s)
    bitcoins = re.findall(bitcoin_pattern, s)
    streets = re.findall(street_pattern, s)
    
    return (emails, ssns, bitcoins, streets)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------

from collections import Counter
def tfidf_data(reviews_ser, review):
    words = re.findall(r'\b\w+\b', review.lower())
    
    word_counts = Counter(words)
    
    total_words = len(words)
    
    tf = {word: count / total_words for word, count in word_counts.items()}
    
    def df(word):
        return reviews_ser.str.contains(r'\b{}\b'.format(re.escape(word)), case=False).sum()
    
    total_docs = len(reviews_ser)
    
    idf = {word: np.log(total_docs / df(word)) for word in word_counts}
    
    tfidf = {word: tf[word] * idf[word] for word in word_counts}
    
    df_tfidf = pd.DataFrame({
        'cnt': word_counts,
        'tf': tf,
        'idf': idf,
        'tfidf': tfidf
    })
    
    return df_tfidf

def relevant_word(out):
    best_word = out['tfidf'].idxmax()
    return best_word


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

def hashtag_list(tweet_text):
    def extract_hashtags(text):
        return re.findall(r'#(\S+)', text)
    
    return tweet_text.apply(extract_hashtags)


def most_common_hashtag(tweet_lists):
    all_hashtags = [hashtag for hashtags in tweet_lists for hashtag in hashtags]
    
    hashtag_freq = pd.Series(all_hashtags).value_counts()
    
    def find_most_common(hashtags):
        if len(hashtags) == 0:
            return pd.NA  # No hashtags in the tweet
        elif len(hashtags) == 1:
            return hashtags[0]  # Only one hashtag in the tweet
        else:
            common_hashtags = hashtag_freq.loc[hashtags]
            return common_hashtags.idxmax()
    
    return tweet_lists.apply(find_most_common)

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def create_features(ira):
    def extract_hashtags(text):
        return re.findall(r'#(\S+)', text)
    
    def most_common_hashtag(tweet_lists):
        all_hashtags = [hashtag for hashtags in tweet_lists for hashtag in hashtags]
        hashtag_freq = pd.Series(all_hashtags).value_counts()
        
        def find_most_common(hashtags):
            if len(hashtags) == 0:
                return pd.NA
            elif len(hashtags) == 1:
                return hashtags[0]
            else:
                common_hashtags = hashtag_freq.loc[hashtags]
                return common_hashtags.idxmax()
        
        return tweet_lists.apply(find_most_common)
    
    def num_hashtags(hashtags):
        return hashtags.apply(len)
    
    def num_tags(tweet_text):
        def extract_tags(text):
            return re.findall(r'@\w+', text)
        return tweet_text.apply(lambda x: len(extract_tags(x)))
    
    def num_links(tweet_text):
        def extract_links(text):
            return re.findall(r'http[s]?://\S+', text)
        return tweet_text.apply(lambda x: len(extract_links(x)))
    
    def is_retweet(tweet_text):
        return tweet_text.apply(lambda x: x.strip().startswith("RT"))
    
    def clean_text(text):
        text = re.sub(r'(RT\s+|@\w+|http[s]?://\S+|#\S+)', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    hashtags = ira['text'].apply(extract_hashtags)
    
    ira_features = pd.DataFrame(index=ira.index)
    
    ira_features['num_hashtags'] = num_hashtags(hashtags)
    ira_features['mc_hashtags'] = most_common_hashtag(hashtags)
    ira_features['num_tags'] = num_tags(ira['text'])
    ira_features['num_links'] = num_links(ira['text'])
    ira_features['is_retweet'] = is_retweet(ira['text'])
    ira_features['text'] = ira['text'].apply(clean_text)
    
    return ira_features[['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']]
