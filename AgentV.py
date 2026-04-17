import pandas as pd, sqlite3, pickle, json, os, matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

class AgentV:
    def __init__(self):
        self.db = 'materials.db'
        self.model_file = 'model_v.pkl'
        self.version_file = 'version.json'
        self.log_file = 'log.csv'
        self.check_file = 'last_count.txt'

    def load_data(self):
        conn = sqlite3.connect(self.db)
        df = pd.read_sql_query("SELECT * FROM mats", conn)
        conn.close()
        return df

    def prepare_features(self, df):
        df['len'] = df['content'].str.len()
        df['words'] = df['content'].str.split().str.len()
        df['has_prev'] = df.get('has_prev', 0)
        df['has_next'] = df.get('has_next', 0)
        df['is_gen'] = df['url'].str.startswith('generated://').astype(int)
        X = df[['len', 'words', 'has_prev', 'has_next', 'is_gen']].fillna(0)
        if 'parallel_cluster' not in df:
            df['parallel_cluster'] = pd.cut(df['len'], bins=3, labels=[0,1,2])
        if 'sequential_cluster' not in df:
            df['sequential_cluster'] = pd.cut(df['len'], bins=3, labels=[0,1,2])
        if 'difficulty_level' not in df:
            df['difficulty_level'] = pd.qcut(df['len'], q=3, labels=['Низкая','Средняя','Высокая'])
        return X, df['parallel_cluster'], df['sequential_cluster'], df['difficulty_level']

    def train_best(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            'RF': RandomForestClassifier(random_state=42),
            'LR': LogisticRegression(max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier()
        }
        best_m, best_name, best_score = None, '', -1
        for name, m in models.items():
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            score = (acc + f1 + rec) / 3
            if score > best_score:
                best_score, best_m, best_name = score, m, name
        print(f"Лучшая модель: {best_name} (score={best_score:.3f})")
        return best_m

    def check_drift(self, new_df):
        if not os.path.exists(self.version_file):
            return False, 0
        with open(self.version_file) as f:
            old = json.load(f)
        old_len = old.get('avg_len', 0)
        new_len = new_df['content'].str.len().mean()
        drift = abs(new_len - old_len) / (old_len + 1e-6)
        return drift > 0.3, drift

    def need_retrain(self, df):
        current_count = len(df)
        if not os.path.exists(self.check_file):
            with open(self.check_file, 'w') as f:
                f.write(str(current_count))
            return True, "Первичное обучение", 0
        with open(self.check_file) as f:
            old_count = int(f.read().strip())
        new_data = current_count > old_count
        drift, score = self.check_drift(df)
        if new_data or drift:
            reason = "Новые данные" if new_data else "Дрейф данных"
            return True, reason, score
        return False, "", 0

    def save_all(self, models, df, reason, drift):
        ver = 1
        if os.path.exists(self.version_file):
            with open(self.version_file) as f:
                ver = json.load(f).get('version', 0) + 1
        data = {
            'version': ver,
            'models': models,
            'avg_len': df['content'].str.len().mean(),
            'features': ['len', 'words', 'has_prev', 'has_next', 'is_gen']
        }
        with open(self.model_file, 'wb') as f:
            pickle.dump(data, f)
        with open(self.version_file, 'w') as f:
            json.dump({'version': ver, 'reason': reason, 'drift': drift,
                       'datetime': datetime.now().isoformat()}, f)
        log = pd.DataFrame([{'version': ver, 'reason': reason, 'drift': drift,
                             'datetime': datetime.now().isoformat()}])
        if os.path.exists(self.log_file):
            log = pd.concat([pd.read_csv(self.log_file), log])
        log.to_csv(self.log_file, index=False)
        with open(self.check_file, 'w') as f:
            f.write(str(len(df)))
        print(f"Версия {ver} сохранена. Причина: {reason}")

    def estimate_time(self, row, diff_model=None):
        words = len(row['content'].split())
        base = words / 200
        if diff_model:
            feats = pd.DataFrame([[len(row['content']), words, row.get('has_prev',0),
                                   row.get('has_next',0), 1 if 'generated://' in row['url'] else 0]],
                                 columns=['len','words','has_prev','has_next','is_gen'])
            diff = diff_model.predict(feats)[0]
            coef = {'Низкая':0.8, 'Средняя':1.0, 'Высокая':1.5}.get(diff, 1.0)
        else:
            coef = 1.0
        return round(base * coef, 1)

    def plot_time(self, df):
        df['time'] = df.apply(self.estimate_time, axis=1)
        df.groupby('subject')['time'].sum().plot(kind='barh')
        plt.title('Время освоения по предметам')
        plt.xlabel('Минуты')
        plt.tight_layout()
        plt.savefig('plots/time.png')
        plt.close()

    def plot_trajectory(self, plan):
        if plan is None or len(plan) == 0:
            return
        plt.figure(figsize=(10, 5))
        plt.barh(range(len(plan)), plan['est_time'].values)
        plt.yticks(range(len(plan)), [f"{s}: {t}" for s,t in zip(plan['subject'], plan['topic'])])
        plt.xlabel('Минуты')
        plt.title('Рекомендуемая последовательность изучения')
        plt.tight_layout()
        plt.savefig('plots/trajectory.png')
        plt.close()

    def run(self):
        df = self.load_data()
        X, y_par, y_seq, y_diff = self.prepare_features(df)
        retrain, reason, drift = self.need_retrain(df)
        if retrain:
            print(f"Переобучение: {reason}")
            models = {
                'parallel': self.train_best(X, y_par),
                'sequential': self.train_best(X, y_seq),
                'difficulty': self.train_best(X, y_diff)
            }
            self.save_all(models, df, reason, drift)
        else:
            with open(self.model_file, 'rb') as f:
                models = pickle.load(f)['models']
            print("Используются существующие модели.")

        self.plot_time(df)
        total = df.apply(lambda r: self.estimate_time(r, models['difficulty']), axis=1).sum()
        print(f"Общее время: {total:.1f} мин ({total/60:.1f} ч)")

        known = input("Изученные темы (через запятую): ").split(',')
        daily = int(input("Минут в день: ") or 60)
        days = int(input("Дней: ") or 30)
        unknown = df[~df['topic'].isin([k.strip() for k in known if k.strip()])].copy()
        unknown['est_time'] = unknown.apply(lambda r: self.estimate_time(r, models['difficulty']), axis=1)
        unknown = unknown.sort_values(['subject', 'has_prev', 'has_next'])
        feasible = unknown[unknown['est_time'].cumsum() <= daily * days]
        self.plot_trajectory(feasible)
        print("План сохранён в plots/trajectory.png")

if __name__ == '__main__':
    AgentV().run()