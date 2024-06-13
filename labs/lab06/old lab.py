# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


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


from bs4 import BeautifulSoup

def extract_book_links(text):
    soup = BeautifulSoup(text, 'lxml')
    books = soup.find_all('article', class_='product_pod')
    urls = []
    for book in books:
        rating = book.find('p', class_='star-rating')['class'][1]
        price = book.find('p', class_='price_color').get_text()
        price = float(price[2:])

        if rating in ['Four','Five'] and price < 50:
            link = book.find('div', class_='image_container').find('a')['href']
            urls.append(link)
        
    return urls

def get_product_info(text, categories):
    keys = ['UPC', 'Product Type', 'Price (excl. tax)', 'Price (incl. tax)', 'Tax', 'Availability', 'Number of reviews', 'Category', 'Rating', 'Description', 'Title']
    soup = BeautifulSoup(text, features="lxml")
    categories = [x.lower() for x in categories]

    this_cat = soup.find('ul', class_='breadcrumb').find_all('a')[-1].text
    if this_cat.lower() not in categories: return None

    first_seven = soup.find_all('td') # Get the first seven values for the dict
    values = [x.get_text() for x in first_seven]
    
    values.append(this_cat) # Add category to dictionary
    
    rating = soup.find('p', class_='star-rating')['class'][1]
    values.append(rating)
    
    description = soup.find('meta', attrs={'name': 'description'})
    description = description['content'].strip()
    values.append(description) 
    
    title = soup.find('title').get_text().strip().split('|')[0].strip()
    values.append(title)
    
    result = dict(zip(keys, values))
    
    return result

def scrape_books(k, categories):
    cols = ['UPC', 'Product Type', 'Price (excl. tax)', 'Price (incl. tax)', 'Tax', 'Availability', 'Number of reviews', 'Category', 'Rating', 'Description', 'Title']
    final = pd.DataFrame(columns=cols)
    for i in range(1, k+1):
        page = requests.get(f'http://books.toscrape.com/catalogue/page-{i}.html')
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
    open_pr = history.iloc[0]['open']
    close_pr = history.iloc[-1]['close']
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


from datetime import datetime

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
    
    site = requests.get(format_url(18344932)).json()
    reverse_ids = site['kids'][::-1]
    # Put all main comments in next_up stack
    for i in reverse_ids:
        next_up.push(i)
    
    # Initialize list for visited comments
    visited = []

    # Loop through next_up and continue adding child comments
    while not next_up.is_empty():
        # move the top comment to the visited list
        top = next_up.pop()
        visited.append(top)
        # Retrieve the children of the comment I just popped 
        try:
            for kid in get_comment(top)['kids']:
                next_up.push(kid)
        # If the comment doesn't have kids, continue
        except KeyError:
            continue
            
    final_df = make_df(visited)
    return final_df