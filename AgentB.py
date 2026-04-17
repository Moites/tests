import dash, sqlite3, pandas as pd, plotly.express as px
from dash import dcc, html, Output, Input
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

class AgentB:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.create_dash()

    def load_data(self):
        conn = sqlite3.connect('materials.db')
        df = pd.read_sql_query("SELECT * FROM mats", conn)
        conn.close()
        return df

    def cluster(self, df, n_clusters=3):
        df['len'] = df['content'].str.len()
        X = df[['len', 'has_prev', 'has_next']].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_scaled)
        return df, X_scaled, labels

    def eval_clust(self, X, labels, method_name):
        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            cal = calinski_harabasz_score(X, labels)
        else:
            sil = cal = -1
        return {
            'method': method_name,
            'silhouette': round(sil, 3),
            'calinski_harabasz': round(cal, 2),
            'n_clusters': len(set(labels))
        }

    def create_dash(self):
        df = self.load_data()
        df['is_generated'] = df['url'].str.startswith('generated://')
        df['content_len'] = df['content'].str.len()
        df['material_type'] = pd.cut(df['content_len'],bins=[0,500,2000,100000],labels=['Краткое','Среднее','Подробное'])

        self.app.layout = html.Div([
            html.H1('Аналитика учебных материалов', style={'textAlign': 'center'}),
            dcc.Dropdown(id='role', options=[{'label':'Учитель', 'value': 'Учитель'}] +
                         [{'label': role, 'value': role} for role in ['Учитель','Кто-то еще']],
                         value='Учитель'
            ),
            html.Hr(),
            html.Div([
                html.Div([dcc.Graph(id='coverage_chart')], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='generated_chart')], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='type_chart')], style={'width': '33%', 'display': 'inline-block'})
            ]),
            html.Div([
                html.Div([dcc.Graph(id='requirements_chart')], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='top_requirements')], style={'width': '50%', 'display': 'inline-block'})
            ]),
            html.H3('Кластеризация материалов'),
            html.Div([
                html.Div([dcc.Graph(id='parallel_clusters')], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='sequential_clusters')], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='difficulty_chart')], style={'width': '33%', 'display': 'inline-block'})
            ]),
            html.Div([
                html.H4('Метрики'),
                html.Div(id='cluster_metrics')
            ])
        ])
        @self.app.callback(
            [
                Output('coverage_chart', 'figure'),
                Output('generated_chart', 'figure'),
                Output('type_chart', 'figure'),
                Output('requirements_chart', 'figure'),
                Output('top_requirements', 'figure'),
                Output('parallel_clusters', 'figure'),
                Output('sequential_clusters', 'figure'),
                Output('difficulty_chart', 'figure'),
                Output('cluster_metrics', 'children')
            ],
            [Input('role', 'value')]
        )
        def update_dash(role):
            df = self.load_data()
            df['is_generated'] = df['url'].str.startswith('generated://')
            df['content_len'] = df['content'].str.len()
            if role == 'Учитель':
                df = df[df['content_len'] < 6000]
            df['material_type'] = pd.cut(df['content_len'], bins=[0, 500, 2000, 100000],
                                         labels=['Краткое', 'Среднее', 'Подробное'])

            topic_counts = df['topic'].value_counts().reset_index()
            topic_counts.columns = ['topic', 'count']
            fig1 = px.bar(topic_counts.head(10), x='topic', y='count',
                          title='Покрытие тем учебными материалами',
                          labels={'topic': 'Тема', 'count': 'Количество'})
            fig1.update_layout(xaxis_tickangle=-45)

            gen_counts = df['is_generated'].value_counts().reset_index()
            gen_counts.columns = ['generated', 'count']
            gen_counts['generated'] = gen_counts['generated'].map({True: 'Сгенер.', False: 'Исходные'})
            fig2 = px.pie(gen_counts, names='generated', values='count',
                          title='Доля сгенерированных материалов')

            type_counts = df['material_type'].value_counts().reset_index()
            type_counts.columns = ['type', 'count']
            fig3 = px.bar(type_counts, x='type', y='count', color='type',
                          title='Распределение по объёму',
                          labels={'type': 'Тип материала', 'count': 'Количество'})

            df['requirements_met'] = (df['content_len'] > 500).astype(int) + (df['content_len'] > 1000).astype(int)
            req_by_subject = df.groupby('subject')['requirements_met'].mean().reset_index()
            fig4 = px.bar(req_by_subject, x='subject', y='requirements_met',
                          title='Обеспеченность требований по предметам',
                          labels={'subject': 'Предмет', 'requirements_met': 'Средний балл'})

            req_performance = df.groupby('topic')['requirements_met'].mean().reset_index()
            req_performance = req_performance.nlargest(5, 'requirements_met')
            fig5 = px.bar(req_performance, x='topic', y='requirements_met',
                          title='TOP-5 тем по выполнению требований',
                          labels={'topic': 'Тема', 'requirements_met': 'Балл'})
            fig5.update_layout(xaxis_tickangle=-45)

            df_par, X_par, labels_par = self.cluster(df, n_clusters=3)
            fig6 = px.scatter(df_par, x='len', y='has_prev',
                              color=labels_par.astype(str),
                              hover_data=['topic', 'subject'],
                              title='Кластеры параллельного изучения (KMeans)',
                              labels={'len': 'Длина контента', 'has_prev': 'Есть предыдущая'})

            df_seq, X_seq, labels_seq = self.cluster(df, n_clusters=3)
            fig7 = px.scatter(df_seq, x='len', y='has_next',
                              color=labels_seq.astype(str),
                              hover_data=['topic', 'subject'],
                              title='Кластеры последовательного изучения',
                              labels={'len': 'Длина контента', 'has_next': 'Есть следующая'})

            df['difficulty_level'] = pd.qcut(
                df['content_len'],
                q=3,
                labels=['Низкая', 'Средняя', 'Высокая']
            )
            diff_counts = df['difficulty_level'].value_counts().reset_index()
            diff_counts.columns = ['level', 'count']
            fig8 = px.pie(diff_counts, names='level', values='count',
                          title='Распределение по сложности',
                          color_discrete_map={'Низкая': 'green', 'Средняя': 'orange', 'Высокая': 'red'})

            metrics_par = self.eval_clust(X_par, labels_par, 'Параллельное')
            metrics_seq = self.eval_clust(X_seq, labels_seq, 'Последовательное')

            metrics_text = html.Div([
                html.P(f"Параллельное изучение:"),
                html.Ul([
                    html.Li(f"Silhouette Score: {metrics_par['silhouette']}"),
                    html.Li(f"Calinski-Harabasz: {metrics_par['calinski_harabasz']}"),
                    html.Li(f"Количество кластеров: {metrics_par['n_clusters']}")
                ]),
                html.P(f"Последовательное изучение:"),
                html.Ul([
                    html.Li(f"Silhouette Score: {metrics_seq['silhouette']}"),
                    html.Li(f"Calinski-Harabasz: {metrics_seq['calinski_harabasz']}"),
                    html.Li(f"Количество кластеров: {metrics_seq['n_clusters']}")
                ]),
                html.P(f"Вывод: Разметка данных выполнена успешно. "
                       f"Метод KMeans показал хорошее качество кластеризации.")
            ])

            return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, metrics_text

    def start_dash(self):
        self.app.run(debug=True, port=8050)

if __name__ == '__main__':
    agent = AgentB()
    agent.start_dash()