from os.path import exists

import pandas as pd, sqlite3, pickle, json, os, matplotlib.pyplot as plt
from datetime import datetime

from matplotlib.pyplot import tight_layout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

class AgentV:
    def __init__(self):
        self.db = 'materials.db'
        self.model_file = 'model.pkl'
        self.version_file = 'version.json'
        self.log_file = 'log.csv'
        self.check_file = 'check.txt'

    def load_data(self):
        conn = sqlite3.connect('materials.db')
        df = pd.read_sql_query("SELECT * FROM mats", conn)
        conn.close()
        return df

    def prepare(self, df):
        df['len'] = df['content'].str.len()
        df['words'] = df['content'].str.split().str.len()
        df['has_prev'] = df.get('has_prev', 0)
        df['has_next'] = df.get('has_next', 0)
        df['is_gen'] = df['url'].str.startswith('generated://').astype(int)
        X = df[['len', 'words', 'has_prev', 'has_next', 'is_gen']].fillna(0)
        if 'par_cluster' not in df:
            df['par_cluster'] = pd.cut(df['len'], bins=3, labels=[0,1,2])
        if 'seq_cluster' not in df:
            df['seq_cluster'] = pd.cut(df['len'], bins=3, labels=[0, 1, 2])
        if 'diff' not in df:
            df['diff'] = pd.cut(df['len'], bins=3, labels=['Низкая','Средняя','Высокая'])
        return X, df['par_cluster'], df['seq_cluster'], df['diff']

    def drift(self, df):
        if not exists(self.version_file):
            return False,0
        with open(self.version_file) as f:
            old = json.load(f)
        old_len = old.get('len', 0)
        new_len = df['content'].str.len().mean()
        drift = abs(new_len - old_len) / (old_len + 1e-6)
        return drift > 0.3 , drift

    def need_retrain(self, df):
        current = len(df)
        if not os.path.exists(self.check_file):
            with open(self.check_file, 'w') as f:
                f.write(str(current))
            return True, 'Первое обучение ', 0
        with open(self.check_file) as f:
            old_count = int(f.read().strip())
        new_data = current > old_count
        drift, score = self.drift(df)
        if new_data or drift:
            reason = 'Новые данные' if new_data else 'Дрейф данных'
            return True, reason, score
        return False, '', 0

    def train_best(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42),
            'KNeighborsClassifier': KNeighborsClassifier()
        }
        best_model, best_name, best_score = None, '', -1
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            score = (accuracy + f1 + recall) / 3
            if score > best_score:
                best_name = name
                best_score = score
                best_model = model
        print(f'Лучшая модель: {best_name}, score: {best_score}')
        return best_model

    def save(self, models, df, reason, drift):
        ver = 1
        if os.path.exists(self.version_file):
            with open(self.version_file) as f:
                ver = json.load(f).get('version', 0) + 1
        data = {
            'version': ver,
            'models': models,
            'features': ['len', 'words', 'has_prev', 'has_next', 'is_gen'],
            'len': df['content'].str.len()
        }
        with open(self.model_file, 'wb') as f:
            pickle.dump(data, f)
        with open(self.version_file, 'w') as f:
            json.dump({'version': ver, 'reason': reason, 'drift': drift,
                         'datetime': datetime.now().isoformat()}, f)
        log = pd.DataFrame([{'version': ver, 'reason': reason, 'drift': drift,
                         'datetime': datetime.now().isoformat()}])
        if os.path.exists(self.log_file):
            log = pd.concat([log, pd.read_csv(self.log_file)])
        log.to_csv(self.log_file, index=False)
        with open(self.check_file, 'w') as f:
            f.write(str(len(df)))
        print('Данные сохраненны')

    def time(self, row, diff = None):
        words = len(row['content'].split())
        base = words / 200
        if diff:
            feats = pd.DataFrame([[len(row['content']), words, row.get('has_prev',0), row.get('has_next',0),
                                   1 if 'generated://' in row['url'] else 0,]],
                                 columns=['len', 'words', 'has_prev', 'has_next', 'is_gen'])
            diff = diff.predict(feats)[0]
            coef = {'Низкая':0.8,'Средняя':1.0,'Высокая':1.5}.get(diff, 1.0)
        else:
            coef = 1.0
        return round(base * coef, 1)

    def plot_time(self, df):
        df['time'] = df.apply(self.time, axis=1)
        df.groupby('subject')['time'].sum().plot(kind='barh')
        plt.xlabel('Минуты')
        plt.tight_layout()
        plt.savefig('time.png')
        plt.show()
        plt.close()

    def run(self):
        df = self.load_data()
        X, y_par, y_seq, y_diff = self.prepare(df)
        retrain, reason, drift = self.need_retrain(df)
        if retrain:
            print('Переобучение: ' + reason)
            models = {
                'parallel': self.train_best(X, y_par),
                'sequential': self.train_best(X, y_seq),
                'difficulty': self.train_best(X, y_diff)
            }
            self.save(models, df, reason, drift)
        else:
            with open(self.model_file, 'rb') as f:
                models = pickle.load(f)['models']

        self.plot_time(df)
