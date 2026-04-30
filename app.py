# ============================================================================
# FLASK ПРИЛОЖЕНИЕ: СИСТЕМА ДИАГНОСТИКИ ЗАБОЛЕВАНИЙ
# ============================================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================================

model = None
label_encoder = None
all_symptoms = None
symptom_translation = {}
russian_to_english = {}
treatment_database = {}

# ============================================================================
# СЛОВАРЬ ПЕРЕВОДА СИМПТОМОВ
# ============================================================================

SYMPTOM_TRANSLATION = {
    'itching': 'зуд', 'skin_rash': 'сыпь', 'fatigue': 'усталость',
    'high_fever': 'высокая температура', 'fever': 'температура',
    'mild_fever': 'небольшая температура', 'headache': 'головная боль',
    'nausea': 'тошнота', 'vomiting': 'рвота', 'loss_of_appetite': 'потеря аппетита',
    'weight_loss': 'потеря веса', 'weight_gain': 'набор веса',
    'abdominal_pain': 'боль в животе', 'stomach_pain': 'боль в желудке',
    'chest_pain': 'боль в груди', 'back_pain': 'боль в спине',
    'neck_pain': 'боль в шее', 'joint_pain': 'боль в суставах',
    'knee_pain': 'боль в колене', 'hip_joint_pain': 'боль в тазобедренном суставе',
    'muscle_pain': 'мышечная боль', 'chills': 'озноб', 'sweating': 'потливость',
    'dizziness': 'головокружение', 'breathlessness': 'одышка',
    'cough': 'кашель', 'diarrhoea': 'диарея', 'constipation': 'запор',
    'dehydration': 'обезвоживание', 'nodal_skin_eruptions': 'узловые высыпания',
    'dischromic_patches': 'дисхромические пятна', 'yellowish_skin': 'желтушность кожи',
    'red_spots_over_body': 'красные пятна по телу', 'blister': 'волдырь',
    'skin_peeling': 'шелушение кожи', 'pus_filled_pimples': 'гнойные прыщи',
    'blackheads': 'угри', 'scurring': 'струпья', 'silver_like_dusting': 'серебристое шелушение',
    'small_dents_in_nails': 'ямки на ногтях', 'inflammatory_nails': 'воспаление ногтей',
    'brittle_nails': 'ломкие ногти', 'puffy_face_and_eyes': 'отёчность лица и глаз',
    'swollen_legs': 'отёк ног', 'swollen_extremeties': 'отёк конечностей',
    'bruising': 'синяки', 'prominent_veins_on_calf': 'выступающие вены на икрах',
    'restlessness': 'беспокойство', 'lethargy': 'вялость', 'depression': 'депрессия',
    'irritability': 'раздражительность', 'anxiety': 'тревожность',
    'blurred_and_distorted_vision': 'нечёткое зрение', 'visual_disturbances': 'нарушения зрения',
    'slurred_speech': 'невнятная речь', 'spinning_movements': 'ощущение вращения',
    'loss_of_balance': 'потеря равновесия', 'unsteadiness': 'неустойчивость',
    'weakness_in_limbs': 'слабость в конечностях', 'weakness_of_one_body_side': 'слабость одной стороны тела',
    'altered_sensorium': 'изменение сознания', 'muscle_weakness': 'мышечная слабость',
    'stiff_neck': 'скованность шеи', 'movement_stiffness': 'скованность движений',
    'painful_walking': 'болезненная ходьба', 'swelling_joints': 'опухание суставов',
    'cramps': 'судороги', 'burning_micturition': 'жжение при мочеиспускании',
    'spotting_urination': 'кровянистые выделения при мочеиспускании',
    'bladder_discomfort': 'дискомфорт в мочевом пузыре', 'foul_smell_of_urine': 'неприятный запах мочи',
    'continuous_feel_of_urine': 'постоянное чувство мочеиспускания', 'polyuria': 'частое мочеиспускание',
    'dark_urine': 'тёмная моча', 'yellow_urine': 'жёлтая моча', 'phlegm': 'мокрота',
    'rusty_sputum': 'ржавая мокрота', 'mucoid_sputum': 'слизистая мокрота',
    'blood_in_sputum': 'кровь в мокроте', 'throat_irritation': 'раздражение горла',
    'throat_pain': 'боль в горле', 'continuous_sneezing': 'непрерывное чихание',
    'runny_nose': 'насморк', 'congestion': 'заложенность носа',
    'sinus_pressure': 'давление в пазухах', 'loss_of_smell': 'потеря обоняния',
    'redness_of_eyes': 'покраснение глаз', 'watering_from_eyes': 'слезотечение',
    'patches_in_throat': 'пятна в горле', 'fast_heart_rate': 'учащённое сердцебиение',
    'palpitations': 'сердцебиение', 'acidity': 'кислотность',
    'indigestion': 'расстройство пищеварения', 'ulcers_on_tongue': 'язвы на языке',
    'passage_of_gases': 'газообразование', 'internal_itching': 'внутренний зуд',
    'swelling_of_stomach': 'отёк желудка', 'distention_of_abdomen': 'вздутие живота',
    'belly_pain': 'боль в животе', 'toxic_look': 'токсический вид',
    'bloody_stool': 'кровавый стул', 'pain_during_bowel_movements': 'боль при дефекации',
    'pain_in_anal_region': 'боль в анальной области', 'irritation_in_anus': 'раздражение в анусе',
    'yellowing_of_eyes': 'пожелтение глаз', 'sunken_eyes': 'запавшие глаза',
    'enlarged_thyroid': 'увеличенная щитовидка', 'swelled_lymph_nodes': 'увеличенные лимфоузлы',
    'malaise': 'недомогание', 'muscle_wasting': 'истощение мышц', 'obesity': 'ожирение',
    'excessive_hunger': 'повышенный голод', 'increased_appetite': 'повышенный аппетит',
    'irregular_sugar_level': 'нерегулярный уровень сахара', 'cold_hands_and_feets': 'холодные руки и ноги',
    'mood_swings': 'перепады настроения', 'abnormal_menstruation': 'нарушение менструации',
    'drying_and_tingling_lips': 'сухость и покалывание губ', 'red_sore_around_nose': 'красная язва вокруг носа',
    'yellow_crust_ooze': 'жёлтая корка с выделениями', 'family_history': 'семейная история',
    'history_of_alcohol_consumption': 'история употребления алкоголя',
    'receiving_blood_transfusion': 'переливание крови', 'receiving_unsterile_injections': 'нестерильные инъекции',
    'extra_marital_contacts': 'внебрачные контакты', 'fluid_overload': 'перегрузка жидкостью',
    'acute_liver_failure': 'острая печёночная недостаточность', 'coma': 'кома',
    'stomach_bleeding': 'желудочное кровотечение', 'lack_of_concentration': 'отсутствие концентрации',
    'loss_of_concentration': 'потеря концентрации', 'pain_behind_the_eyes': 'боль за глазами',
    'swollen_blood_vessels': 'набухшие сосуды', 'shivering': 'дрожь'
}

RUSSIAN_TO_ENGLISH = {v: k for k, v in SYMPTOM_TRANSLATION.items()}

# ============================================================================
# БАЗА ДАННЫХ ЛЕЧЕНИЯ
# ============================================================================

TREATMENT_DATABASE = {
    'Fungal infection': {'name': 'Грибковая инфекция',
                         'treatments': ['Противогрибковые препараты (клотримазол, флуконазол)',
                                        'Противогрибковые кремы и мази', 'Гигиена поражённых участков',
                                        'Избегание влаги и потливости'],
                         'prevention': 'Держать кожу сухой, носить дышащую одежду'},
    'Allergy': {'name': 'Аллергия',
                'treatments': ['Антигистаминные препараты (цетиризин, лоратадин)', 'Избегание аллергена',
                               'Кортикостероидные кремы'], 'prevention': 'Выявить и исключить аллерген'},
    'GERD': {'name': 'Гастроэзофагеальная рефлюксная болезнь',
             'treatments': ['Ингибиторы протонной помпы (омепразол)', 'Антациды',
                            'Диета: избегать острой и жирной пищи'], 'prevention': 'Питание небольшими порциями'},
    'Diabetes': {'name': 'Диабет', 'treatments': ['Инсулин или пероральные препараты', 'Контроль уровня глюкозы',
                                                  'Диета с низким содержанием углеводов',
                                                  'Регулярная физическая активность'],
                 'prevention': 'Здоровое питание, регулярные упражнения'},
    'Gastroenteritis': {'name': 'Гастроэнтерит',
                        'treatments': ['Регидратация (обильное питьё)', 'Пробиотики', 'Противодиарейные препараты'],
                        'prevention': 'Гигиена рук, безопасная пища и вода'},
    'Bronchial Asthma': {'name': 'Бронхиальная астма',
                         'treatments': ['Ингаляторы (сальбутамол)', 'Кортикостероидные ингаляторы', 'Бронходилататоры'],
                         'prevention': 'Избегать аллергены, дым, холодный воздух'},
    'Hypertension': {'name': 'Гипертония',
                     'treatments': ['Антигипертензивные препараты', 'Диета DASH', 'Регулярные упражнения',
                                    'Контроль веса'], 'prevention': 'Здоровый образ жизни'},
    'Migraine': {'name': 'Мигрень',
                 'treatments': ['Триптаны (суматриптан)', 'Обезболивающие (ибупрофен)', 'Противорвотные препараты'],
                 'prevention': 'Избегать триггеры, регулярный сон'},
    'Common Cold': {'name': 'Простуда', 'treatments': ['Отдых', 'Обильное питьё', 'Жаропонижающие', 'Витамины'],
                    'prevention': 'Гигиена рук, избегание контактов'},
    'Pneumonia': {'name': 'Пневмония', 'treatments': ['Антибиотики', 'Госпитализация', 'Кислородная терапия'],
                  'prevention': 'Вакцинация, отказ от курения'},
    'Chicken pox': {'name': 'Ветряная оспа',
                    'treatments': ['Противовирусные препараты', 'Антигистаминные от зуда', 'Изоляция'],
                    'prevention': 'Вакцинация'},
    'Dengue': {'name': 'Лихорадка денге',
               'treatments': ['Госпитализация', 'Внутривенная регидратация', 'Обезболивающие'],
               'prevention': 'Защита от комаров'},
    'Typhoid': {'name': 'Брюшной тиф', 'treatments': ['Антибиотики', 'Регидратация', 'Диета'],
                'prevention': 'Вакцинация, гигиена'},
    'Hepatitis B': {'name': 'Гепатит B', 'treatments': ['Противовирусные препараты', 'Мониторинг печени'],
                    'prevention': 'Вакцинация'},
    'Tuberculosis': {'name': 'Туберкулёз',
                     'treatments': ['Комбинация антибиотиков (6+ месяцев)', 'Изониазид, рифампицин'],
                     'prevention': 'Вакцинация БЦЖ'},
    'Heart attack': {'name': 'Инфаркт', 'treatments': ['ЭКСТРЕННАЯ ГОСПИТАЛИЗАЦИЯ', 'Аспирин', 'Тромболитики'],
                     'prevention': 'Контроль давления, холестерина'},
    'Acne': {'name': 'Акне', 'treatments': ['Бензоилпероксид', 'Ретиноиды', 'Гигиена кожи'],
             'prevention': 'Гигиена, не трогать лицо'},
    'Psoriasis': {'name': 'Псориаз', 'treatments': ['Кортикостероидные кремы', 'Фототерапия', 'Увлажнение кожи'],
                  'prevention': 'Избегать стресс'},
    'Urinary tract infection': {'name': 'Инфекция мочевыводящих путей',
                                'treatments': ['Антибиотики', 'Обильное питьё', 'Пробиотики'],
                                'prevention': 'Гигиена, обильное питьё'},
    'Arthritis': {'name': 'Артрит', 'treatments': ['НПВС', 'Физиотерапия', 'Упражнения'],
                  'prevention': 'Упражнения, здоровый вес'},
    'Hypothyroidism': {'name': 'Гипотиреоз', 'treatments': ['Левотироксин', 'Мониторинг ТТГ', 'Диета'],
                       'prevention': 'Регулярные проверки'},
    'Hyperthyroidism': {'name': 'Гипертиреоз', 'treatments': ['Антитиреоидные препараты', 'Бета-блокаторы'],
                        'prevention': 'Регулярные проверки'},
    'Jaundice': {'name': 'Желтуха', 'treatments': ['Лечение основной причины', 'Гидратация', 'Диета для печени'],
                 'prevention': 'Вакцинация от гепатита'},
    'Malaria': {'name': 'Малярия', 'treatments': ['Противомалярийные препараты', 'Поддерживающая терапия'],
                'prevention': 'Защита от комаров'},
    'AIDS': {'name': 'ВИЧ/СПИД', 'treatments': ['Антиретровирусная терапия (АРТ)', 'Профилактика инфекций'],
             'prevention': 'Безопасные контакты'},
    'Drug Reaction': {'name': 'Лекарственная реакция', 'treatments': ['Прекратить приём препарата', 'Антигистаминные'],
                      'prevention': 'Сообщать о реакциях'},
    'Hypoglycemia': {'name': 'Гипогликемия', 'treatments': ['Быстрые углеводы', 'Глюкоза', 'Регулярное питание'],
                     'prevention': 'Регулярное питание'},
    'Varicose veins': {'name': 'Варикоз', 'treatments': ['Компрессионные чулки', 'Упражнения', 'Хирургия'],
                       'prevention': 'Упражнения, контроль веса'},
    'Dimorphic hemmorhoids(piles)': {'name': 'Геморрой',
                                     'treatments': ['Кремы от геморроя', 'Диета с клетчаткой', 'Обильное питьё'],
                                     'prevention': 'Диета, регулярный стул'},
    'Cervical spondylosis': {'name': 'Шейный спондилёз', 'treatments': ['НПВС', 'Физиотерапия', 'Упражнения для шеи'],
                             'prevention': 'Правильная осанка'},
    'Paralysis (brain hemorrhage)': {'name': 'Паралич',
                                     'treatments': ['Экстренная госпитализация', 'Реабилитация', 'Физиотерапия'],
                                     'prevention': 'Контроль давления'},
    'Chronic cholestasis': {'name': 'Хронический холестаз',
                            'treatments': ['Урсодезоксихолевая кислота', 'Витамины A, D, E, K'],
                            'prevention': 'Обследования печени'},
    'Peptic ulcer diseae': {'name': 'Язвенная болезнь',
                            'treatments': ['Ингибиторы протонной помпы', 'Антибиотики', 'Диета'],
                            'prevention': 'Избегать НПВС, алкоголь'},
    'Alcoholic hepatitis': {'name': 'Алкогольный гепатит',
                            'treatments': ['Отказ от алкоголя', 'Кортикостероиды', 'Витамины'],
                            'prevention': 'Отказ от алкоголя'},
    'Hepatitis A': {'name': 'Гепатит A', 'treatments': ['Поддерживающая терапия', 'Отдых', 'Гидратация'],
                    'prevention': 'Вакцинация, гигиена'},
    'Hepatitis C': {'name': 'Гепатит C', 'treatments': ['Противовирусные препараты', 'Мониторинг печени'],
                    'prevention': 'Стерильные инструменты'},
    'Hepatitis D': {'name': 'Гепатит D', 'treatments': ['Интерферон', 'Лечение гепатита B'],
                    'prevention': 'Вакцинация от гепатита B'},
    'Hepatitis E': {'name': 'Гепатит E', 'treatments': ['Поддерживающая терапия', 'Гидратация'],
                    'prevention': 'Безопасная вода'},
    'Osteoarthristis': {'name': 'Остеоартрит', 'treatments': ['НПВС', 'Физиотерапия', 'Контроль веса'],
                        'prevention': 'Упражнения, контроль веса'},
    '(vertigo) Paroymsal  Positional Vertigo': {'name': 'Головокружение',
                                                'treatments': ['Манёвр Эпли', 'Вестибулярная реабилитация'],
                                                'prevention': 'Осторожные движения'},
    'Impetigo': {'name': 'Импетиго', 'treatments': ['Антибиотические мази', 'Гигиена', 'Изоляция'],
                 'prevention': 'Гигиена'}
}


# ============================================================================
# ФУНКЦИИ ЗАГРУЗКИ И ОБУЧЕНИЯ МОДЕЛИ
# ============================================================================

def load_or_train_model():
    """Загрузка обученной модели или обучение новой"""
    global model, label_encoder, all_symptoms

    model_path = 'models/disease_model.keras'
    encoder_path = 'models/disease_model_encoder.pkl'
    symptoms_path = 'models/disease_model_symptoms.json'

    # Проверяем наличие сохранённой модели
    if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(symptoms_path):
        print("Загрузка сохранённой модели...")
        model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        with open(symptoms_path, 'r', encoding='utf-8') as f:
            all_symptoms = json.load(f)
        print("Модель загружена успешно!")
        return True

    # Обучение новой модели
    print("Обучение новой модели...")
    return train_new_model()


def train_new_model():
    """Обучение новой нейросети"""
    global model, label_encoder, all_symptoms

    # Загрузка данных
    df = pd.read_csv('DiseaseAndSymptoms.csv')
    df['Disease'] = df['Disease'].str.strip()

    # Получение всех симптомов
    all_symptoms_set = set()
    for col in df.columns[1:]:
        df[col] = df[col].fillna('').str.strip()
        all_symptoms_set.update(df[col].unique())
    all_symptoms_set.discard('')
    all_symptoms = list(all_symptoms_set)

    print(f"Найдено {len(all_symptoms)} симптомов и {df['Disease'].nunique()} заболеваний")

    # Создание матрицы симптомов
    symptom_matrix = np.zeros((len(df), len(all_symptoms)))
    for idx, row in df.iterrows():
        for col in df.columns[1:]:
            symptom = row[col]
            if symptom and symptom in all_symptoms:
                symptom_idx = all_symptoms.index(symptom)
                symptom_matrix[idx, symptom_idx] = 1

    # Кодирование болезней
    label_encoder = LabelEncoder()
    disease_labels = label_encoder.fit_transform(df['Disease'])

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        symptom_matrix, disease_labels, test_size=0.2, random_state=42, stratify=disease_labels
    )

    # Создание модели
    model = keras.Sequential([
        layers.Input(shape=(len(all_symptoms),)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Обучение
    print("Обучение модели...")
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Оценка
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Точность: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Сохранение
    os.makedirs('models', exist_ok=True)
    model.save('models/disease_model.keras')
    with open('models/disease_model_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('models/disease_model_symptoms.json', 'w', encoding='utf-8') as f:
        json.dump(all_symptoms, f, ensure_ascii=False)

    print("Модель сохранена!")
    return True


def predict_disease(symptoms_input, top_k=3):
    """Предсказание болезни по симптомам"""
    symptom_vector = np.zeros((1, len(all_symptoms)))

    for symptom in symptoms_input:
        symptom_eng = RUSSIAN_TO_ENGLISH.get(symptom.lower().strip(), symptom.lower().strip())
        if symptom_eng in all_symptoms:
            idx = all_symptoms.index(symptom_eng)
            symptom_vector[0, idx] = 1

    predictions = model.predict(symptom_vector, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:top_k]

    results = []
    for idx in top_indices:
        disease_name = label_encoder.inverse_transform([idx])[0]
        confidence = float(predictions[idx] * 100)
        results.append({
            'disease_eng': disease_name,
            'confidence': round(confidence, 2)
        })

    return results


def get_treatment(disease_eng):
    """Получение рекомендаций по лечению"""
    if disease_eng in TREATMENT_DATABASE:
        return TREATMENT_DATABASE[disease_eng]
    for key in TREATMENT_DATABASE:
        if disease_eng.lower() in key.lower() or key.lower() in disease_eng.lower():
            return TREATMENT_DATABASE[key]
    return None


# ============================================================================
# FLASK МАРШРУТЫ
# ============================================================================

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """Получение всех доступных симптомов на русском"""
    symptoms_ru = sorted(list(set(SYMPTOM_TRANSLATION.values())))
    return jsonify({
        'success': True,
        'symptoms': symptoms_ru,
        'count': len(symptoms_ru)
    })


@app.route('/api/symptoms/categories', methods=['GET'])
def get_symptom_categories():
    """Получение симптомов по категориям"""
    categories = {
        'general': ['зуд', 'сыпь', 'усталость', 'высокая температура', 'температура', 'небольшая температура',
                    'головная боль', 'тошнота', 'рвота', 'потеря аппетита', 'потеря веса', 'набор веса',
                    'боль в животе', 'боль в желудке', 'боль в груди', 'боль в спине', 'боль в шее',
                    'боль в суставах', 'мышечная боль', 'озноб', 'потливость', 'головокружение',
                    'одышка', 'кашель', 'диарея', 'запор', 'обезвоживание'],
        'skin': ['узловые высыпания', 'дисхромические пятна', 'желтушность кожи', 'красные пятна по телу',
                 'волдырь', 'шелушение кожи', 'гнойные прыщи', 'угри', 'струпья', 'серебристое шелушение',
                 'ямки на ногтях', 'воспаление ногтей', 'ломкие ногти', 'отёчность лица и глаз',
                 'отёк ног', 'отёк конечностей', 'синяки'],
        'neuro': ['беспокойство', 'вялость', 'депрессия', 'раздражительность', 'тревожность',
                  'нечёткое зрение', 'нарушения зрения', 'невнятная речь', 'ощущение вращения',
                  'потеря равновесия', 'неустойчивость', 'слабость в конечностях', 'мышечная слабость',
                  'скованность шеи', 'скованность движений', 'болезненная ходьба', 'судороги'],
        'digestive': ['кислотность', 'расстройство пищеварения', 'язвы на языке', 'газообразование',
                      'внутренний зуд', 'отёк желудка', 'вздутие живота', 'токсический вид',
                      'кровавый стул', 'боль при дефекации', 'боль в анальной области', 'раздражение в анусе'],
        'respiratory': ['мокрота', 'ржавая мокрота', 'слизистая мокрота', 'кровь в мокроте',
                        'раздражение горла', 'боль в горле', 'непрерывное чихание', 'насморк',
                        'заложенность носа', 'давление в пазухах', 'потеря обоняния', 'покраснение глаз',
                        'слезотечение', 'учащённое сердцебиение', 'сердцебиение']
    }
    return jsonify({'success': True, 'categories': categories})


@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """Диагностика по симптомам"""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])

        if not symptoms:
            return jsonify({'success': False, 'error': 'Симптомы не указаны'}), 400

        predictions = predict_disease(symptoms)

        results = []
        for pred in predictions:
            treatment = get_treatment(pred['disease_eng'])
            results.append({
                'disease': treatment['name'] if treatment else pred['disease_eng'],
                'disease_eng': pred['disease_eng'],
                'confidence': pred['confidence'],
                'treatments': treatment['treatments'] if treatment else [],
                'prevention': treatment['prevention'] if treatment else 'Регулярные медицинские осмотры'
            })

        return jsonify({
            'success': True,
            'predictions': results,
            'input_symptoms': symptoms
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/treatment/<disease>', methods=['GET'])
def get_treatment_api(disease):
    """Получение рекомендаций по лечению для болезни"""
    treatment = get_treatment(disease)
    if treatment:
        return jsonify({'success': True, 'treatment': treatment})
    return jsonify({'success': False, 'error': 'Лечение не найдено'}), 404


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Статус модели"""
    return jsonify({
        'success': True,
        'model_loaded': model is not None,
        'symptoms_count': len(all_symptoms) if all_symptoms else 0,
        'diseases_count': len(label_encoder.classes_) if label_encoder else 0
    })


# ============================================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("СИСТЕМА ДИАГНОСТИКИ ЗАБОЛЕВАНИЙ НА ОСНОВЕ ИИ")
    print("=" * 70)

    # Загрузка или обучение модели
    load_or_train_model()

    print("\nЗапуск Flask-сервера...")
    print("Откройте в браузере: http://localhost:5000")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5000)