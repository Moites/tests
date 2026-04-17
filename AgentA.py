import requests, sqlite3, os, re, time
from bs4 import BeautifulSoup
from datetime import datetime
from gigachat import GigaChat

db = 'materials.db'
urls = [
    'https://metanit.com/sharp/windowsforms/1.1.php',
    'https://metanit.com/sharp/windowsforms/1.2.php',
    'https://metanit.com/sharp/windowsforms/1.3.php',
    'https://metanit.com/sharp/windowsforms/2.1.php',
    'https://metanit.com/sharp/windowsforms/2.2.php',
    'https://metanit.com/sharp/windowsforms/2.3.php',
    'https://metanit.com/sharp/windowsforms/2.4.php',
    'https://metanit.com/sharp/windowsforms/2.5.php',
    'https://metanit.com/sharp/windowsforms/3.1.php',
    'https://metanit.com/sharp/windowsforms/3.2.php',
    'https://metanit.com/sharp/windowsforms/3.3.php',
    'https://metanit.com/sharp/windowsforms/3.4.php',
    'https://metanit.com/sharp/windowsforms/3.5.php',
    'https://metanit.com/sharp/windowsforms/4.1.php',
    'https://metanit.com/sharp/windowsforms/4.2.php',
    'https://metanit.com/sharp/windowsforms/4.3.php',
    'https://metanit.com/sharp/windowsforms/4.4.php',
    'https://metanit.com/sharp/windowsforms/4.5.php',
    'https://metanit.com/sharp/windowsforms/4.6.php',
    'https://metanit.com/sharp/windowsforms/4.7.php',
    'https://metanit.com/sharp/windowsforms/4.8.php',
    'https://metanit.com/sharp/windowsforms/4.9.php',
    'https://metanit.com/sharp/windowsforms/4.10.php',
    'https://metanit.com/sharp/windowsforms/4.11.php',
    'https://metanit.com/sharp/windowsforms/4.12.php',
    'https://metanit.com/sharp/windowsforms/4.13.php',
    'https://metanit.com/sharp/windowsforms/4.14.php',
    'https://metanit.com/sharp/windowsforms/4.15.php',
    'https://metanit.com/sharp/windowsforms/4.16.php',
    'https://metanit.com/sharp/windowsforms/4.17.php',
    'https://metanit.com/sharp/windowsforms/4.18.php',
    'https://metanit.com/sharp/windowsforms/4.19.php',
    'https://metanit.com/sharp/windowsforms/4.20.php',
    'https://metanit.com/sharp/windowsforms/4.22.php',
]
giga = os.getenv('GIGACHAT_CREDENTIALS',
                       'MDE5ZDg3NzYtMTY5OS03M2IwLWFmODQtMGMxYzI3ZmI0MmE5OmNmMjUxM2QzLWE2M2UtNDdkNy1hNzM0LWY2ODdhZDZmMzlmNg==')

def init_db():
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT, topic TEXT, content TEXT,
        annotation TEXT, url TEXT UNIQUE, conclusion TEXT,
        datetime TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_mat(d):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO mats 
        (subject,topic,content,annotation,url,conclusion,datetime)
        VALUES (?,?,?,?,?,?,?)''', (
        d['subject'], d['topic'], d['content'], d['annotation'],
        d['url'], d['conclusion'], d['datetime']))
    conn.commit()
    conn.close()

def url_exists(url):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('SELECT 1 FROM mats WHERE url=?', (url,))
    r = c.fetchone()
    conn.close()
    return r is not None

def get_topics():
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('SELECT topic FROM mats ORDER BY topic')
    t = [r[0] for r in c.fetchall()]
    conn.close()
    return t

def fetch(url):
    try:
        r = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if r.status_code == 200:
            return r.text
    except Exception as e:
        print(f'Ошибка {url}: {e}')
    return None

def get_subject(text):
    client = GigaChat(credentials=giga, verify_ssl_certs=False)
    prompt = (
        f"Из следующего текста извлеки только название дисциплины (предмета).\n"
        f"Ответ должен состоять только из названия дисциплины, без пояснений.\n"
        f"Примеры ответов: C#, математика, русский язык.\n"
        f"Текст: {text}"
    )
    resp = client.chat(prompt)
    content = resp.choices[0].message.content.strip()
    return content

def parse(html, url):
    if not html: return None
    soup = BeautifulSoup(html, 'html.parser')
    topic = None
    h1 = soup.find('h1')
    if h1 and h1.get_text(strip=True):
        topic = h1.get_text(strip=True)
    if not topic:
        title = soup.find('title')
        if title:
            title_text = title.get_text(strip=True)
            topic = re.sub(r'\s*[-|]\s*METANIT\.COM.*$', '', title_text, flags=re.I).strip()
            if not topic or len(topic) < 3:
                topic = None
    if not topic:
        breadcrumb = soup.find('div', class_='breadcrumb') or soup.find('nav', class_='breadcrumb')
        if breadcrumb:
            links = breadcrumb.find_all('a')
            if links:
                topic = links[-1].get_text(strip=True)
    if not topic or topic == 'Без названия':
        url_parts = url.split('/')
        filename = url_parts[-1].replace('.php', '')
        if re.match(r'\d+\.\d+', filename):
            topic = f"Раздел {filename}"
        else:
            topic = f"Материал по Windows Forms ({filename})"
    content = soup.find('div', class_='content') or soup.find('main') or soup
    for t in content(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        t.decompose()

    text = '\n'.join([p.get_text(' ', strip=True) for p in content.find_all(['p', 'h2', 'h3', 'li', 'pre']) if
                      len(p.get_text(strip=True)) > 20])
    if len(text) < 100: return None

    annot = text[:200] + '...' if len(text) > 200 else text

    return {
        'subject': get_subject(text), 'topic': topic, 'content': text,
        'annotation': annot, 'url': url,
        'conclusion': '',
        'datetime': datetime.now()
    }

def check(content):
    if len(content) < 200: return 'НЕДОПУСТИМО: короткий'
    if len(content) > 100000: return 'НЕДОПУСТИМО: длинный'
    return 'ДОПУСТИМО'

def generate(topic):
    try:
        client = GigaChat(credentials=giga, verify_ssl_certs=False)
        prompt = f'Создай учебный материал по теме "{topic}". Объём 500-1000 слов, Структура: введение, теория, пример, выводы.'
        resp = client.chat(prompt)
        content = resp.choices[0].message.content.strip()

        return {
            'subject': get_subject(content), 'topic': topic, 'content': content,
            'annotation': content[:200] + '...' if len(content) > 200 else content,
            'url': f'generated://{topic}',
            'conclusion': check(content),
            'datetime': datetime.now()
        }
    except Exception as e:
        print(f'Ошибка генерации: {e}')
        return None

def process(urls):
    print(f'Обработка {len(urls)} ссылок')
    for url in urls:
        if url_exists(url):
            print(f'Пропущено: {url}')
            continue
        print(f'{url}')
        html = fetch(url)
        if not html: continue
        d = parse(html, url)
        if not d: continue
        d['conclusion'] = check(d['content'])
        save_mat(d)
        print(f'{d["topic"]}')
        time.sleep(1)

def add_adjacent_flags():
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(mats)")
    cols = [col[1] for col in cur.fetchall()]
    if 'has_prev' not in cols:
        cur.execute('ALTER TABLE mats ADD COLUMN has_prev INTEGER DEFAULT 0')
    if 'has_next' not in cols:
        cur.execute('ALTER TABLE mats ADD COLUMN has_next INTEGER DEFAULT 0')
    cur.execute('''SELECT id, subject, topic FROM mats 
                 WHERE subject IS NOT NULL AND topic IS NOT NULL
                 ORDER BY subject, topic''')
    rows = cur.fetchall()
    by_subject = {}
    for id_, subject, topic in rows:
        by_subject.setdefault(subject, []).append((id_, topic))
    for ids_topics in by_subject.values():
        for i, (id_, _) in enumerate(ids_topics):
            has_prev = 1 if i > 0 else 0
            has_next = 1 if i < len(ids_topics) - 1 else 0
            cur.execute('UPDATE mats SET has_prev=?, has_next=? WHERE id=?',
                      (has_prev, has_next, id_))
    conn.commit()
    conn.close()

def run():
    init_db()
    process(urls)

    while True:
        cmd = input('\nТема для генерации (exit): ').strip()
        if cmd.lower() in 'exit': break
        if cmd:
            m = generate(cmd)
            if m and 'НЕДОПУСТИМО' not in m['conclusion']:
                save_mat(m)
                print(f'Сгенерировано: {cmd}')
    add_adjacent_flags()

if __name__ == '__main__':
    run()