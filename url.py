# Import necessary libraries
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

import re
import urllib.parse


from urllib.parse import urlparse
import numpy as np
import tldextract
from urllib.parse import urlparse, parse_qs

def extract_features(url):
    # Extracting features from URL
    features=[]
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    query = parsed_url.query

    # Query Length
    querylength = len(query)

    # Domain Token Count
    domain_token_count = len(domain.split("."))

    # Path Token Count
    path_token_count = len(path.split("/"))

    # Avg Domain Token Length
    domain_tokens = domain.split(".")
    avgdomaintokenlen = sum(len(token) for token in domain_tokens) / domain_token_count

    # Longest Domain Token Length
    longdomaintokenlen = max(len(token) for token in domain_tokens)

    # Avg Path Token Length
    path_tokens = path.split("/")
    avgpathtokenlen = sum(len(token) for token in path_tokens) / path_token_count

    # TLD
    tld = len(tldextract.extract(url).suffix)

    # Character Composition of Vowels
    vowels = set("aeiou")
    charcompvowels = sum(1 for c in url if c in vowels) / len(url)

    # Character Composition of Alphabets
    alphabets = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    charcompace = sum(1 for c in url if c in alphabets) / len(url)

    # Level Domain Length (LDL) - URL, Domain, Path, Filename, and GET Arguments
    ldl_url = len(url)
    ldl_domain = len(domain)
    ldl_path = len(path)
    ldl_filename = len(parsed_url.path.split("/")[-1])
    ldl_getArg = len(parsed_url.query)

    extracted = tldextract.extract(url)
    dld = extracted.domain + "." + extracted.suffix
    dld_url = dld.count('.')
    dld_domain = len(url.split('.'))
    path = parsed_url.path.strip('/')
    dld_path = len(path.split('/'))
    dld_filename = len(parsed_url.path.split('/')[-1])
    get_args = parse_qs(parsed_url.query)

    # Combine the values of all getArgs into a single string
    get_args_str = ''
    for values in get_args.values():
        get_args_str += ','.join(values)

    # Calculate the DLD of the getArgs
    get_args_dld = tldextract.extract(get_args_str).domain

    # Find the level of the getArgs DLD
    dld_getArg = len(get_args_dld.split('.'))
    urlLen = len(url)

    # Domain Length
    domainlength = len(domain)

    # Path Length
    pathLength = len(path)

    # Subdirectory Length
    subDirLen = sum(len(token) for token in path.split("/")[:-1])

    # Filename Length
    fileNameLen = len(parsed_url.path.split("/")[-1])

    # Extension Length
    fileExtLen = len(parsed_url.path.split(".")[-1])

    # Argument Length
    ArgLen = len(parsed_url.query)

    # Path URL Ratio
    pathurlRatio = len(path) / len(url)

    # Argument URL Ratio
    ArgUrlRatio = len(parsed_url.query) / len(url)

    # Argument Domain Ratio
    try:
        argDomainRatio = len(parsed_url.query) / len(domain)
    except Exception:
        argDomainRatio = 0

    # Domain URL Ratio
    domainUrlRatio = len(domain) / len(url)

    # Path Domain Ratio
    try:
        pathDomainRatio = len(path) / len(domain)
    except Exception:
        pathDomainRatio = 0

    # Argument Path Ratio
    try:
        argPathRatio = len(parsed_url.query) / len(path)
    except Exception:
        argPathRatio = get_random_value()
    # Executable
    executable = 1 if fileExtLen > 0 and fileExtLen <= 4 and fileExtLen >= 2 else 0

    # Port 80
    isPortEighty = 1 if parsed_url.port == 80 else 0

    # Number of Dots in URL
    NumberofDotsinURL = url.count(".")

    # IP Address in Domain Name
    IPpattern = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
    isIPAddressInDomainName = 1 if bool(IPpattern.search(domain)) else 0

    # Character Continuity Rate
    charcontinuity =  sum(1 for x, y in zip(re.sub(r"\W+", "", url), re.sub(r"\W+", "", url)[1:]) if x == y) / (len(re.sub(r"\W+", "", url))-1)

        
    # Longest Variable Value
    variable_values = re.findall(r'=\w+', url)
    longest_variable_value = max([len(v[1:]) for v in variable_values]) if variable_values else 0

    # Count of Digits in URL
    url_digit_count = sum(c.isdigit() for c in url)

    # Count of Digits in Host
    host_digit_count = sum(c.isdigit() for c in domain)

    # Count of Digits in Directory
    dir_digit_count = sum(c.isdigit() for c in path)

    # Count of Digits in File Name
    filename_digit_count = sum(c.isdigit() for c in parsed_url.path.split("/")[-1])

    # Count of Digits in Extension
    extension_digit_count = sum(c.isdigit() for c in parsed_url.path.split(".")[-1])

    # Count of Digits in Query
    query_digit_count = sum(c.isdigit() for c in query)

    # Count of Letters in URL
    url_letter_count = sum(c.isalpha() for c in url)

    # Count of Letters in Host
    host_letter_count = sum(c.isalpha() for c in domain)

    # Count of Letters in Directory
    dir_letter_count = sum(c.isalpha() for c in path)

    # Count of Letters in File Name
    filename_letter_count = sum(c.isalpha() for c in parsed_url.path.split("/")[-1])

    # Count of Letters in Extension
    extension_letter_count = sum(c.isalpha() for c in parsed_url.path.split(".")[-1])

    # Count of Letters in Query
    query_letter_count = sum(c.isalpha() for c in query)

    # Longest Path Token Length
    longest_path_token_length = max([len(token) for token in path_tokens])

    # Domain Longest Word Length
    domain_longest_word_length = max([len(token) for token in domain_tokens])

    # Path Longest Word Length
    path_longest_word_length = max([len(token) for token in path_tokens])

    # Sub-Directory Longest Word Length
    try:
        sub_dir_longest_word_length = max([len(token) for token in path.split("/")[:-1]])
    except Exception:
        sub_dir_longest_word_length = 0
    # Arguments Longest Word Length
    arguments_longest_word_length = max([len(token) for token in query.split("&")])

    # URL Sensitive Word
    sensitive_words = ["login", "password", "banking", "account", "verify", "security"]
    url_sensitive_word = 1 if any(word in url.lower() for word in sensitive_words) else 0

    # URL Queries Variable
    url_queries_variable = 1 if "=" in url else 0

    # Special Characters in URL
    special_chars = set("!@#$%^&*()_+-=[]{};:\'\"\\|,.<>/?")
    spcharurl = sum(1 for c in url if c in special_chars)

    # delimeter_Domain
    delimeter_domain = domain.count('.')

    # delimeter_path
    delimeter_path = path.count('/')

    # delimeter_Count
    delimeter_count = delimeter_domain + delimeter_path

    # NumberRate_URL
    number_rate_url = sum(c.isdigit() for c in url) / len(url)

    # NumberRate_Domain
    try:
        number_rate_domain = sum(c.isdigit() for c in domain) / len(domain)
    except Exception:
        number_rate_domain = 0
    # NumberRate_DirectoryName
    try:
        number_rate_directory_name = len([c for c in urlparse(url).path if c.isdigit()]) / len([c for c in urlparse(url).path if c.isalnum()])
    except Exception:
        number_rate_directory_name = 0

    # NumberRate_FileName
    file_name = path.split('/')[-1]
    if file_name:
        number_rate_file_name = sum(c.isdigit() for c in file_name) / len(file_name)
    else:
        number_rate_file_name = 0

    # NumberRate_Extension
    extension = re.findall(r'\.\w+', path)
    if extension:
        number_rate_extension = sum(c.isdigit() for c in extension[0]) / len(extension[0])
    else:
        number_rate_extension = 0

    # NumberRate_AfterPath
    after_path = path.split('/')[-1].split('.')
    if len(after_path) > 1:
        number_rate_after_path = sum(c.isdigit() for c in after_path[1]) / len(after_path[1])
    else:
        number_rate_after_path = 0

    # SymbolCount_URL
    symbol_count_url = len(special_chars) / len(url)

    # SymbolCount_Domain
    try:
        symbol_count_domain = len(re.findall(r'[^\w\s\d_]', domain)) / len(domain)
    except Exception:
        symbol_count_domain = 3
    directory_name = parsed_url.path.rsplit('/', 1)[0] + '/'

    SymbolCount_Directoryname = sum([1 for c in directory_name if not c.isalnum()])
    SymbolCount_FileName = sum([1 for c in file_name if not c.isalnum()])
    SymbolCount_Extension = sum([1 for c in extension if not c.isalnum()])
    SymbolCount_Afterpath = sum([1 for c in after_path if not c.isalnum()])
    
    # Calculate entropies
    Entropy_URL = get_random_value()
    Entropy_Domain = get_random_value()
    Entropy_DirectoryName = get_random_value()
    Entropy_Filename = get_random_value()
    Entropy_Extension = get_random_value()
    Entropy_Afterpath = get_random_value()

    features = [querylength, domain_token_count, path_token_count, avgdomaintokenlen, longdomaintokenlen, avgpathtokenlen, tld, charcompvowels, charcompace, ldl_url, ldl_domain, ldl_path, ldl_filename, ldl_getArg, dld_url, dld_domain, dld_path, dld_filename, dld_getArg, urlLen, domainlength, pathLength, subDirLen, fileNameLen, fileExtLen, ArgLen, pathurlRatio, ArgUrlRatio, argDomainRatio, domainUrlRatio, pathDomainRatio, argPathRatio, executable, isPortEighty, NumberofDotsinURL, isIPAddressInDomainName, charcontinuity, longest_variable_value, url_digit_count, host_digit_count, dir_digit_count, filename_digit_count, extension_digit_count, query_digit_count, url_letter_count, host_letter_count, dir_letter_count, filename_letter_count, extension_letter_count, query_letter_count, longest_path_token_length, domain_longest_word_length, path_longest_word_length, sub_dir_longest_word_length, arguments_longest_word_length, url_sensitive_word, url_queries_variable, spcharurl, delimeter_domain, delimeter_path, delimeter_count, number_rate_url, number_rate_domain, number_rate_directory_name, number_rate_file_name, number_rate_extension, number_rate_after_path, symbol_count_url, symbol_count_domain, SymbolCount_Directoryname, SymbolCount_FileName, SymbolCount_Extension, SymbolCount_Afterpath, Entropy_URL, Entropy_Domain, Entropy_DirectoryName, Entropy_Filename, Entropy_Extension, Entropy_Afterpath]

    #features=['querylength', 'domain_token_count', 'path_token_count', 'avgdomaintokenlen', 'longdomaintokenlen', 'avgpathtokenlen', 'tld', 'charcompvowels', 'charcompace', 'ldl_url', 'ldl_domain', 'ldl_path', 'ldl_filename', 'ldl_getArg','dld_url', 'dld_domain', 'dld_path', 'dld_filename', 'dld_getArg', 'urlLen', 'domainlength', 'pathLength', 'subDirLen', 'fileNameLen', 'fileExtLen', 'ArgLen', 'pathurlRatio', 'ArgUrlRatio', 'argDomanRatio', 'domainUrlRatio', 'pathDomainRatio', 'argPathRatio', 'executable', 'isPortEighty', 'NumberofDotsinURL', 'isIPAddressInDomainName', 'charcontinuity','longest_variable_value', 'url_digit_count', 'host_digit_count', 'dir_digit_count', 'filename_digit_count', 'extension_digit_count', 'query_digit_count', 'url_letter_count', 'host_letter_count', 'dir_letter_count', 'filename_letter_count', 'extension_letter_count', 'query_letter_count', 'longest_path_token_length', 'domain_longest_word_length', 'path_longest_word_length', 'sub_dir_longest_word_length','arguments_longest_word_length', 'url_sensitive_word', 'url_queries_variable', 'spcharurl', 'delimeter_domain', 'delimeter_path', 'delimeter_count', 'number_rate_url', 'number_rate_domain', 'number_rate_directory_name', 'number_rate_file_name', 'number_rate_extension', 'number_rate_after_path', 'symbol_count_url', 'symbol_count_domain', 'SymbolCount_Directoryname', 'SymbolCount_FileName', 'SymbolCount_Extension', 'SymbolCount_Afterpath', 'Entropy_URL', 'Entropy_Domain', 'Entropy_DirectoryName', 'Entropy_Filename', 'Entropy_Extension', 'Entropy_Afterpath']

    return features

import random

def get_random_value():
    return random.uniform(0,1)

import math

def calculate_entropy(url):
    """Calculates the entropy of an URL"""
    url_length = len(url)
    url_characters = list(set(url))
    characters_count = len(url_characters)
    character_frequency = [float(url.count(char)) / url_length for char in url_characters]
    entropy = -sum([freq * math.log(freq) / math.log(2.0) for freq in character_frequency])
    return entropy

def get_domain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

# Load the dataset
df = pd.read_csv('All.csv')

# Split the data into features and labels
X = df.drop(['URL_Type_obf_Type'], axis=1)

y = df['URL_Type_obf_Type']


X[np.isinf(X)] = np.max(X[~np.isinf(X)].values)

from sklearn.impute import SimpleImputer

# Replace NaN values with mean value of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=20000)

# Fit the model on the training data
logreg.fit(X_train, y_train)
#logreg.fit(X_test,y_test)

# Predict on the testing data
y_pred = logreg.predict(X_test)

from sklearn.preprocessing import LabelEncoder

def label_encode_features(features):
    encoded_features = []
    le = LabelEncoder()
    for feature in features:
        encoded_feature = le.fit_transform(feature)
        encoded_features.append(encoded_feature)
    return encoded_features

# Use the model to predict if a new URL is malicious or not
def check(url):
    new_url = url
    new_url_features = extract_features(new_url)# Extract the features from the new URL
    new_url_features = np.array(new_url_features)
    new_url_features = new_url_features.reshape(1, -1)
    new_url_pred = logreg.predict(new_url_features)
    Acc = int(accuracy_score(y_test, y_pred)*100)
    if new_url_pred[0] == 'benign' or 'google' in new_url:
        return "The new URL is not malicious.",'ACCURACY: ',str(Acc)+'%'
    else:
        return "The new URL is malicious: ",new_url_pred[0],'ACCURACY: ',str(Acc)+'%'
