from bs4 import BeautifulSoup
import requests
import re
import os


def scrape_articles():
    # 1. Get list of latest bitcoin article urls
    cd_url = "https://www.coindesk.com"
    source = requests.get(cd_url).text
    soup = BeautifulSoup(source, 'lxml')

    raw_links = []
    for line in soup.find_all('div', class_='articles-holder'):
        raw_links.append(line.find_all('a', class_="headline", href=True))

    btc_links = []
    for url in raw_links:
        btc_links.append(re.findall(r'href=\"(.+?)\"', str(url)))

    btc_links = [y for x in btc_links for y in x if y is not None and re.search(r'\bbitcoin\b', str(y))]

    # 2. Visit urls and scrape article bodies
    dir = "Articles\\"
    for i, link in enumerate(btc_links):
        source = requests.get(cd_url + link).text
        soup = BeautifulSoup(source, 'lxml')
        with open(f"{dir}article{i}.txt", 'w', encoding='utf-8') as txt_file:
            for tag in soup.find('div', {'class': 'main-body-grid'}).find_all('p'):
                tag = str(tag)
                if re.search(r'\bRead more\b', tag) or re.search(r'\bDISCLOSURE\b', tag):
                    break
                tag = re.sub('<.*?>', '', tag)
                txt_file.write(tag + '\n')


# count files in dir, split into training and testing and return two lists of filepaths
def split_data(dir):
    all_files = []
    total_files = 0
    for _, _, fname in os.walk(dir):
        for txt in fname:
            all_files.append(dir + txt)
            total_files += 1
    training_total = int(total_files * 0.8)
    training_files = all_files[0:training_total]
    test_files = all_files[training_total:]
    return training_files, test_files


# convert article txt files to lists
def txt_to_list(dir, num=13):
    str_list = []
    for i in range(num):
        with open(dir + f'article{i}.txt', encoding='utf-8') as fname:
            for line in fname:
                str_list.append([line])
    return str_list


# prepare new data
def prepare_new_data(str_list):
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(str_list)
        test_sequences = tokenizer.texts_to_sequences(str_list)
    padded_sequences = pad_sequences(test_sequences, maxlen=150)
    return padded_sequences


# evaluate model
model = tf.keras.models.load_model(r'C:\Users\Sean-work\OneDrive\Coding\PycharmProjects\NLP_btcsentanalysis\model1')
model.predict(str_list)
