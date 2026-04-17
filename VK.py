import requests
import time
import vk_api

TOKEN = ("vk1.a.bdufNq57BBWBVJQl0QZ_8BThy26jE7JYpx6kOieckk4veXg7FQRZcpVAT59gIq_xkkRahSCk13h1vxUP5z5E7eP_pZghwOJzNXs6ht"
"5EYm5fn--mREJVF3mmDjUzfb6ONTCsF1kEjSsdZdONVxPvjudM3NMXlzEU3J6olu8UjsHm9yqUE2IRpAxQTrDF9WxBvcOkXSP_BFhJf1NWFEW9eA")
API_URL = "http://localhost:8000"

vk = vk_api.VkApi(token=TOKEN).get_api()
last_id = 0

def call_api(endpoint, data=None):
    try:
        if data:
            resp = requests.post(f"{API_URL}{endpoint}", json=data)
        else:
            resp = requests.get(f"{API_URL}{endpoint}")
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

def format_analysis(result):
    return f"""
Анализ материала:
• Тема: {result.get('topic', '—')}
• Статус: {result.get('check', '—')}
• Сложность: {result.get('difficulty', '—')}
• Время освоения: ~{result.get('est_time_min', 0)} мин

Кластеры:
• Параллельный: {result.get('parallel_cluster', '—')}
• Последовательный: {result.get('sequential_cluster', '—')}
"""

def format_plan(plan_data):
    total = plan_data['total_available_min']
    planned = plan_data['planned_min']

    text = f"""
План обучения:
• Доступно времени: {total} мин ({total//60} ч {total%60} мин)
• Запланировано: {planned} мин
• Материалов: {plan_data['materials_count']}

Рекомендуемый порядок:
"""
    for i, item in enumerate(plan_data['plan'][:10], 1):
        text += f"\n{i}. [{item['subject']}] {item['topic'][:40]} — {item['est_time']} мин"
    if len(plan_data['plan']) > 10:
        text += f"\n... и ещё {len(plan_data['plan']) - 10} материалов"
    return text

print("Бот ВК запущен\n")
COMMANDS = """
Доступные команды:
• /help — это сообщение
• /subjects — список предметов
• /topics [предмет] — темы по предмету
• /check [текст] — проверка материала
• /plan [изученные через запятую] [мин/день] [дней] — план обучения
Пример: /plan Python,SQL 60 30
"""
while True:
        for item in vk.messages.getConversations(count=5)['items']:
            msg = item['last_message']
            if msg['out'] or msg['id'] <= last_id:
                continue
            last_id = msg['id']
            peer_id = msg['peer_id']
            text = msg.get('text', '').strip()
            print(f"\n{peer_id}: {text[:50]}")
            response = None
            if text.startswith('/help') or text.lower() == 'привет':
                response = f"Привет! Я бот для анализа учебных материалов.\n{COMMANDS}"
            elif text.startswith('/subjects'):
                data = call_api("/subjects")
                response = "Предметы:\n• " + "\n• ".join(data['subjects'])

            elif text.startswith('/topics'):
                parts = text.split(maxsplit=1)
                subject = parts[1] if len(parts) > 1 else None
                data = call_api(f"/topics?subject={subject}" if subject else "/topics")
                response = f"Темы{f' по {subject}' if subject else ''}:\n• " + "\n• ".join(data['topics'][:20])

            elif text.startswith('/check'):
                content = text[7:].strip()
                if content:
                    data = call_api("/analyze", {"content": content, "topic": "Проверка"})
                    response = format_analysis(data)
                else:
                    response = "Отправь текст после /check"

            elif text.startswith('/plan'):
                parts = text[6:].strip().split()
                known = []
                daily = 60
                days = 30
                if parts:
                    known = [k.strip() for k in parts[0].split(',') if k.strip()]
                if len(parts) > 1:
                    daily = int(parts[1]) if parts[1].isdigit() else 60
                if len(parts) > 2:
                    days = int(parts[2]) if parts[2].isdigit() else 30

                data = call_api("/trajectory", {
                    "known_topics": known,
                    "daily_minutes": daily,
                    "deadline_days": days
                })
                response = format_plan(data)

            else:
                if len(text) > 50:
                    data = call_api("/analyze", {"content": text, "topic": "Из сообщения"})
                    if data:
                        response = format_analysis(data)
                    else:
                        response = COMMANDS
                else:
                    response = COMMANDS
            if response:
                vk.messages.send(
                    peer_id=peer_id,
                    message=response,
                    random_id=int(time.time() * 1000)
                )
                print("Ответ отправлен")