# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def extract_book_links(text):
    # parse html
    soup = bs4.BeautifulSoup(text, features="lxml")
    
    # Find all articles with class "product_pod"
    books = soup.find_all('article', class_='product_pod')
    
    # list to store the URLs
    book_urls = []
    
#     price_check = []
    # iterate through each book
    for book in books:
        # Find the star-rating element
        star_rating = book.find('p', class_='star-rating')
        
        # find price
        price = book.find('p', class_='price_color').get_text()
        # get rid of pound sign
        price = price[2:]
        price = float(price)
        
        
        # check star rating and price
#         if star_rating and 'Four' in star_rating['class'] and price < 50:
        if star_rating and price < 50:
            if 'Five' in star_rating['class'] or 'Four' in star_rating['class']:
                link = book.find('div', class_='image_container').find('a')['href']
                # append url to list
                book_urls.append(link)
    
    return book_urls


def get_product_info(text, categories):
    keys = ['UPC', 'Product Type', 'Price (excl. tax)', 'Price (incl. tax)', 'Tax', 'Availability', 'Number of reviews', 'Category', 'Rating', 'Description', 'Title']
    info_soup = bs4.BeautifulSoup(text, features="lxml")
    categories = [x.lower() for x in categories]

    cat = info_soup.find('ul', class_='breadcrumb')
    cat = cat.find_all('a')[-1].text
    if cat.lower() not in categories:
        return None

    # now make the dictionary
    first_seven = info_soup.find_all('td')
    values = [x.get_text() for x in first_seven]
    values.append(cat)
    rating = info_soup.find('p', class_='star-rating')
    class_value = rating['class'][1]
    values.append(class_value)
    description = info_soup.find('meta', attrs={'name': 'description'})
    description = description['content'].strip()
    values.append(description)
    title = info_soup.find('title').get_text().strip()
    title = title.split('|')
    title = title[0].strip()
    values.append(title)
    result = dict(zip(keys, values))
    return result

def scrape_books(k, categories):
    cols = ['UPC', 'Product Type', 'Price (excl. tax)', 'Price (incl. tax)', 'Tax', 'Availability', 'Number of reviews', 'Category', 'Rating', 'Description', 'Title']
    final = pd.DataFrame(columns=cols)
    for i in range(1, k + 1):
        page = requests.get(f"http://books.toscrape.com/catalogue/page-{i}.html")
        page_txt = page.text
        links = extract_book_links(page_txt)
        for book in links:
            info = requests.get(f'https://books.toscrape.com/catalogue/{book}')
            info_dict = get_product_info(info.text, categories)  
            if info_dict == None:
                continue
            info_df = pd.DataFrame([info_dict])
            final = pd.concat([final, info_df])
    return final
    

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def stock_history(ticker, year, month):
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{pd.Period(start_date).days_in_month}"

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey=ATgVbtI43Q4WNnBNUqiuy0VaNjyyqu7o"
    response = requests.get(url)
    data = response.json()

    historical_data = data['historical']
    df = pd.DataFrame(historical_data)

    return df

def stock_stats(history):
    open_pr = history.iloc[-1]['open']
    close_pr = history.iloc[0]['close']
    percent_change = ((close_pr - open_pr) / open_pr) * 100

    total_vol = 0
    for index, row in history.iterrows():
        avg_pr = (row['high'] + row['low']) / 2
        daily_vol = row['volume'] * avg_pr
        total_vol += daily_vol
    total_vol_B = total_vol / 1e9

    percent_change = f"{percent_change:+.2f}%"
    total_vol_B = f"{total_vol_B:.2f}B"

    return (percent_change, total_vol_B)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

def format_url(code):
    url = f'https://hacker-news.firebaseio.com/v0/item/{code}.json'
    return url

def get_comment(code):
    comment = requests.get(format_url(code)).json()
    return comment

# class for stack
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        raise IndexError("pop from an empty stack")

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

    
from datetime import datetime    
    
def make_df(visited):
    keys = ['id', 'by', 'text', 'parent', 'time']
    result = pd.DataFrame(columns=keys) 
    dead_alive = []
    for i in visited:
        attributes = []
        com = get_comment(i)
        try: 
            dead_alive.append(com['dead'])
        except KeyError:
            dead_alive.append(False)
        for i in keys:
            try:
                attributes.append(com[i])
            except KeyError:
                attributes.append(np.NaN)

        result.loc[len(result)] = attributes
    dead_series = pd.Series(dead_alive)
    filtered = result[~dead_series].reset_index(drop=True)
    
    filtered['time'] = filtered['time'].apply(lambda x: datetime.fromtimestamp(x))
    return filtered
    


def get_comments(storyid):
    next_up = Stack()
    
    site = requests.get(format_url(storyid)).json()
    reverse_ids = site['kids'][::-1]
    # put all the main comments into the next_up stack
    for i in reverse_ids:
        next_up.push(i)
    
    # initialize list for visited comments
    visited = []

    # loop through next_up and continue adding child comments
    while not next_up.is_empty():
        # move the top comment to the visited list
        top = next_up.pop()
        visited.append(top)
        # get the children of the comment we just popped 
        try:
            for kid in get_comment(top)['kids']:
                next_up.push(kid)
        # if the comment doesn't have any kids, just continue
        except KeyError:
            continue
            
    final_df = make_df(visited)
    return final_df

