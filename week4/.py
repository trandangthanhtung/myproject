import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


vnexpress_categories = {
    'thoi-su': 'Thời sự',
    'the-gioi': 'Thế giới',
    'kinh-doanh': 'Kinh doanh',
    'khoa-hoc': 'Công nghệ',
    'the-thao': 'Thể thao',
    'bat-dong-san': 'Bất động sản',
    'suc-khoe': 'Sức khỏe',
    'giai-tri': 'Giải trí',
    'phap-luat': 'Pháp luật',
    'giao-duc': 'Giáo dục',
    'doi-song': 'Đời sống',
    'oto-xe-may': 'Xe',
    'du-lich': 'Du lịch',
    'y-kien': 'Ý kiến',
    'tam-su': 'Tâm sự'
}

vietnamnet_categories = {
    'thoi-su': 'Thời sự',
    'the-gioi': 'Thế giới',
    'kinh-doanh': 'Kinh doanh',
    'cong-nghe': 'Công nghệ',
    'the-thao': 'Thể thao',
    'bat-dong-san': 'Bất động sản',
    'suc-khoe': 'Sức khỏe',
    'giai-tri': 'Giải trí',
    'phap-luat': 'Pháp luật',
    'giao-duc': 'Giáo dục',
    'doi-song': 'Đời sống',
    'oto-xe-may': 'Xe',
    'du-lich': 'Du lịch',
    'ban-doc': 'Ý kiến',
}

tuoitre_categories = {
    'thoi-su': 'Thời sự',
    'the-gioi': 'Thế giới',
    'kinh-doanh': 'Kinh doanh',
    'cong-nghe': 'Công nghệ',
    'the-thao': 'Thể thao',
    'bat-dong-san': 'Bất động sản',
    'suc-khoe': 'Sức khỏe',
    'giai-tri': 'Giải trí',
    'phap-luat': 'Pháp luật',
    'giao-duc': 'Giáo dục',
    'doi-song': 'Đời sống',
    'xe': 'Xe',
    'du-lich': 'Du lịch',
    'ban-doc': 'Ý kiến',
}

dantri_categories = {
    'thoi-su': 'Thời sự',
    'the-gioi': 'Thế giới',
    'kinh-doanh': 'Kinh doanh',
    'suc-khoe': 'Sức khỏe',
    'giai-tri': 'Giải trí',
    'the-thao': 'Thể thao',
    'giao-duc-huong-nghiep': 'Giáo dục',
    'bat-dong-san': 'Bất động sản',
    'o-to-xe-may': 'Xe',
    'du-lich': 'Du lịch',
    'cong-nghe': 'Công nghệ',
    'doi-song': 'Đời sống',
    'phap-luat': 'Pháp luật',
}



def crawl_vnexpress():
    articles = []
    for slug, category in vnexpress_categories.items():
        print(f'Crawling VNExpress: {category}')
        for page in range(1, 5):  
            if page == 1:
                url = f'https://vnexpress.net/{slug}'
            else:
                url = f'https://vnexpress.net/{slug}-p{page}'
            try:
                resp = requests.get(url)
                soup = BeautifulSoup(resp.content, 'html.parser')
                news = soup.select('h3.title-news a')

                for item in news:
                    link = item['href']
                    try:
                        art_resp = requests.get(link)
                        art_soup = BeautifulSoup(art_resp.content, 'html.parser')
                        title = art_soup.find('h1', class_='title-detail').text.strip()
                        content = ' '.join([p.text.strip() for p in art_soup.select('p.Normal')])

                        articles.append({
                            'title': title,
                            'content': content,
                            'category': category,
                            'source': 'VNExpress'
                        })
                        time.sleep(0.3)
                    except:
                        continue
            except:
                continue
    return articles

def crawl_vietnamnet():
    articles = []
    for slug, category in vietnamnet_categories.items():
        print(f'Crawling Vietnamnet: {category}')
        for page in range(1, 5):
            if page == 1:
                url = f'https://vietnamnet.vn/{slug}'
            else:
                url = f'https://vietnamnet.vn/{slug}/trang{page}.html'
            try:
                resp = requests.get(url)
                soup = BeautifulSoup(resp.content, 'html.parser')
                news = soup.select('h3.title a')

                for item in news:
                    link = item['href']
                    if not link.startswith('http'):
                        link = 'https://vietnamnet.vn' + link
                    try:
                        art_resp = requests.get(link)
                        art_soup = BeautifulSoup(art_resp.content, 'html.parser')
                        title = art_soup.find('h1', class_='main-title').text.strip()
                        content = ' '.join([p.text.strip() for p in art_soup.select('div.maincontent p')])

                        articles.append({
                            'title': title,
                            'content': content,
                            'category': category,
                            'source': 'Vietnamnet'
                        })
                        time.sleep(0.3)
                    except:
                        continue
            except:
                continue
    return articles

def crawl_tuoitre():
    articles = []
    for slug, category in tuoitre_categories.items():
        print(f'Crawling Tuổi Trẻ: {category}')
        for page in range(1, 5):
            if page == 1:
                url = f'https://tuoitre.vn/{slug}.htm'
            else:
                url = f'https://tuoitre.vn/{slug}-trang-{page}.htm'
            try:
                resp = requests.get(url)
                soup = BeautifulSoup(resp.content, 'html.parser')
                news = soup.select('h3.name-news a')

                for item in news:
                    link = item['href']
                    if not link.startswith('http'):
                        link = 'https://tuoitre.vn' + link
                    try:
                        art_resp = requests.get(link)
                        art_soup = BeautifulSoup(art_resp.content, 'html.parser')
                        title = art_soup.find('h1', class_='article-title').text.strip()
                        content = ' '.join([p.text.strip() for p in art_soup.select('div#main-detail-body p')])

                        articles.append({
                            'title': title,
                            'content': content,
                            'category': category,
                            'source': 'Tuổi Trẻ'
                        })
                        time.sleep(0.3)
                    except:
                        continue
            except:
                continue
    return articles

def crawl_dantri():
    articles = []
    for slug, category in dantri_categories.items():
        print(f'Crawling Dân Trí: {category}')
        for page in range(1, 5):
            if page == 1:
                url = f'https://dantri.com.vn/{slug}.htm'
            else:
                url = f'https://dantri.com.vn/{slug}/trang-{page}.htm'
            try:
                resp = requests.get(url)
                soup = BeautifulSoup(resp.content, 'html.parser')
                news = soup.select('h3.article-title a')

                for item in news:
                    link = item['href']
                    if not link.startswith('http'):
                        link = 'https://dantri.com.vn' + link
                    try:
                        art_resp = requests.get(link)
                        art_soup = BeautifulSoup(art_resp.content, 'html.parser')
                        title = art_soup.find('h1', class_='title-page detail').text.strip()
                        content = ' '.join([p.text.strip() for p in art_soup.select('div.singular-content p')])

                        articles.append({
                            'title': title,
                            'content': content,
                            'category': category,
                            'source': 'Dân trí'
                        })
                        time.sleep(0.3)
                    except:
                        continue
            except:
                continue
    return articles

# -------------------- MAIN FUNCTION ---------------------

def crawl_all_news():
    all_articles = []
    all_articles += crawl_vnexpress()
    all_articles += crawl_vietnamnet()
    all_articles += crawl_tuoitre()
    all_articles += crawl_dantri()
    return all_articles

# -------------------- RUN & SAVE ---------------------

if __name__ == "__main__":
    data = crawl_all_news()
    df = pd.DataFrame(data)
    df.to_csv('news_dataset_full.csv', index=False, encoding='utf-8-sig')
    print(f"Done! Collected {len(df)} articles.") 
