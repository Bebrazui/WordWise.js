'use server';
/**
 * @fileOverview A contextual response AI bot.
 *
 * - contextualResponse - A function that handles the response generation process.
 * - ContextualResponseInput - The input type for the contextualResponse function.
 * - ContextualResponseOutput - The return type for the contextualResponse function.
 */

import {z} from 'zod';
import { findBestMatch } from '@/lib/semantic-search';

const ContextualResponseInputSchema = z.object({
  userInput: z
    .string()
    .describe('The user input to which the AI should respond.'),
});
export type ContextualResponseInput = z.infer<
  typeof ContextualResponseInputSchema
>;

const ContextualResponseOutputSchema = z.object({
  aiResponse: z.string().describe('The AI response based on the user input.'),
});
export type ContextualResponseOutput = z.infer<
  typeof ContextualResponseOutputSchema
>;

// --- Start of the bot's "brain" ---

// Levenshtein distance function for typo correction
function levenshteinDistance(s1: string, s2: string): number {
    s1 = s1.toLowerCase();
    s2 = s2.toLowerCase();

    const costs = [];
    for (let i = 0; i <= s1.length; i++) {
        let lastValue = i;
        for (let j = 0; j <= s2.length; j++) {
            if (i === 0) {
                costs[j] = j;
            } else {
                if (j > 0) {
                    let newValue = costs[j - 1];
                    if (s1.charAt(i - 1) !== s2.charAt(j - 1)) {
                        newValue = Math.min(Math.min(newValue, lastValue), costs[j]) + 1;
                    }
                    costs[j - 1] = lastValue;
                    lastValue = newValue;
                }
            }
        }
        if (i > 0) {
            costs[s2.length] = lastValue;
        }
    }
    return costs[s2.length];
}

function correctSpelling(word: string, vocabulary: string[]): string {
    if (vocabulary.includes(word)) {
        return word;
    }

    let minDistance = Infinity;
    let bestMatch = word;
    
    const threshold = word.length > 5 ? 2 : 1;

    for (const vocabWord of vocabulary) {
        const distance = levenshteinDistance(word, vocabWord);
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = vocabWord;
        }
    }

    return minDistance <= threshold ? bestMatch : word;
}

const vocabulary: string[] = [
    'привет', 'здравствуй', 'добрый день', 'добрый вечер', 'доброе утро', 'пока', 'до свидания', 'увидимся', 'рад был пообщаться',
    'как дела', 'что нового', 'как ты', 'что делаешь', 'чем занимаешься', 'расскажи о себе', 'кто ты', 'как настроение', 'что ты умеешь', 'ты бот', 'твоя личность', 'твоя технология',
    'хорошо', 'отлично', 'неплохо', 'так себе', 'замечательно', 'нормально', 'я в порядке', 'все хорошо', 'бывает и лучше',
    'радость', 'грусть', 'удивление', 'скука', 'интерес', 'счастье', 'злость', 'спокойствие', 'вдохновение', 'любопытство',
    'я', 'ты', 'он', 'она', 'мы', 'вы', 'они', 'человек', 'друг', 'программист', 'собеседник', 'бот', 'искусственный интеллект',
    'думать', 'говорить', 'писать', 'читать', 'смотреть', 'слушать', 'работать', 'учиться', 'помогать', 'знать', 'любить', 'хотеть', 'мочь', 'создавать', 'улучшать',
    'думаю', 'думаешь', 'думает', 'думаем', 'думаете', 'думают',
    'говорю', 'говоришь', 'говорит', 'говорим', 'говорите', 'говорят',
    'пишу', 'пишешь', 'пишет', 'пишем', 'пишете', 'пишут',
    'знаю', 'знаешь', 'знает', 'знаем', 'знаете', 'знают',
    'могу', 'можешь', 'может', 'можем', 'можете', 'могут',
    'хочу', 'хочешь', 'хочет', 'хотим', 'хотите', 'хотят',
    'работаю', 'работаешь', 'работает', 'работаем', 'работаете', 'работают',
    'учусь', 'учишься', 'учится', 'учимся', 'учитесь', 'учатся',
    'делаю', 'делаешь', 'делает', 'делаем', 'делаете', 'делают',
    'отвечаю', 'отвечаешь', 'отвечает', 'отвечаем', 'отвечаете', 'отвечают',
    'слово', 'слова', 'фраз', 'фраза', 'предложение', 'текст', 'код', 'программа', 'идея', 'мысль', 'вопрос', 'ответ', 'мир', 'жизнь', 'время', 'работа', 'компьютер', 'информация',
    'интересный', 'скучный', 'новый', 'старый', 'хороший', 'плохой', 'большой', 'маленький', 'умный', 'простой', 'сложный', 'красивый', 'важный', 'логичный',
    'сегодня', 'завтра', 'вчера', 'всегда', 'иногда', 'никогда', 'здесь', 'там', 'очень', 'немного', 'быстро', 'медленно', 'почему', 'зачем', 'как',
    'и', 'а', 'но', 'или', 'потому что', 'если', 'что', 'чтобы', 'в', 'на', 'о', 'про', 'с', 'к', 'по', 'из', 'для',
    'дом', 'город', 'страна', 'интернет', 'реальность', 'виртуальность',
    'алгоритм', 'данные', 'сеть', 'база данных', 'интерфейс', 'разработка', 'тестирование', 'облако', 'нейросеть', 'машинное обучение', 'цепи Маркова',
    'смысл', 'сознание', 'бытие', 'знание', 'реальность', 'свобода', 'выбор', 'цель',
    'музыка', 'кино', 'книга', 'искусство', 'наука', 'путешествие', 'спорт', 'игра',
    'погода', 'солнце', 'дождь', 'небо', 'звезда', 'планета', 'природа',
    'ну', 'эм', 'вот', 'кстати', 'знаешь', 'хм', 'понимаешь',
    'анализировать', 'строить', 'генерировать', 'отвечать', 'предполагать', 'советовать', 'обсуждать', 'запоминать',
    'язык', 'модель', 'вероятность', 'статистика', 'контекст', 'диалог', 'цель', 'задача', 'общение', 'юмор',
    'спасибо', 'благодарю', 'извини', 'прости', 'да', 'конечно', 'согласен', 'нет', 'не согласен',
    'расскажи шутку', 'пошути', 'у тебя есть чувства', 'ты умный', 'что такое искусственный интеллект', 'расскажи про нейросети',
    'в чем смысл жизни', 'что такое счастье', 'что такое любовь', 'что ты думаешь о', 'тебе нравится', 'что лучше',
    'какое у тебя хобби', 'ты любишь музыку', 'посоветуй книгу', 'какой фильм посмотреть', 'который час', 'сколько времени',
    'какая сегодня погода', 'прогноз погоды',
    'система', 'анализ', 'эксперимент', 'теория', 'гипотеза', 'открытие', 'исследование', 'робот', 'автоматизация', 'процессор', 'память', 'устройство', 'гаджет', 'приложение', 'сайт', 'пользователь', 'безопасность', 'шифрование',
    'этика', 'эстетика', 'концепция', 'парадокс', 'сомнение', 'уверенность', 'гармония', 'хаос', 'структура', 'причина', 'следствие', 'сущность', 'явление', 'абстракция', 'метафора', 'символ', 'аналогия', 'принцип', 'закон', 'правило',
    'тревога', 'страх', 'надежда', 'отчаяние', 'эмпатия', 'симпатия', 'антипатия', 'удовольствие', 'разочарование', 'удовлетворение', 'замешательство', 'ностальгия', 'меланхолия', 'эйфория', 'апатия', 'волнение',
    'театр', 'живопись', 'скульптура', 'архитектура', 'дизайн', 'фотография', 'поэзия', 'проза', 'автор', 'художник', 'композитор', 'режиссер', 'актер', 'стиль', 'жанр', 'шедевр', 'критика', 'ритм', 'мелодия', 'образ', 'сюжет', 'персонаж',
    'дискуссия', 'спор', 'аргумент', 'мнение', 'факт', 'точка зрения', 'компромисс', 'сотрудничество', 'конфликт', 'общество', 'культура', 'традиция', 'история', 'политика', 'экономика', 'норма', 'ценность', 'личность', 'группа', 'команда', 'семья', 'враг', 'знакомый',
    'качество', 'количество', 'свойство', 'характеристика', 'особенность', 'деталь', 'аспект', 'фактор', 'результат', 'последствие', 'преимущество', 'недостаток', 'возможность', 'ограничение', 'проблема', 'решение', 'ситуация', 'окружение', 'вселенная', 'космос', 'галактика',
    'марс', 'температура', 'спутник', 'фобос', 'деймос', 'медузы'
];

const synonymLibrary: { [key: string]: string[] } = {
    'привет': ['здравствуй', 'добрый день', 'добрый вечер'],
    'пока': ['до свидания', 'увидимся', 'прощай'],
    'хорошо': ['отлично', 'замечательно', 'прекрасно', 'в порядке'],
    'плохо': ['не очень', 'так себе', 'неважно'],
    'думать': ['размышлять', 'считать', 'полагать', 'анализировать'],
    'говорить': ['сказать', 'сообщать', 'произносить', 'отвечать'],
    'делать': ['выполнять', 'создавать', 'совершать'],
    'знать': ['понимать', 'быть в курсе', 'осознавать'],
    'помочь': ['оказать содействие', 'поддержать', 'быть полезным'],
    'интересный': ['увлекательный', 'любопытный', 'занимательный', 'необычный'],
    'человек': ['индивид', 'личность', 'персона'],
    'бот': ['программа', 'собеседник', 'искусственный интеллект', 'ИИ', 'виртуальный ассистент'],
    'задача': ['цель', 'миссия', 'задание', 'вопрос'],
    'ответ': ['реплика', 'реакция', 'решение'],
    'вопрос': ['запрос', 'обращение', 'интерес'],
    'проблема': ['сложность', 'затруднение', 'задача'],
    'решение': ['выход', 'ответ', 'способ'],
    'красивый': ['прекрасный', 'восхитительный', 'эстетичный'],
    'важный': ['значимый', 'существенный', 'ключевой'],
    'простой': ['легкий', 'элементарный', 'понятный'],
    'сложный': ['трудный', 'непростой', 'запутанный'],
    'создавать': ['генерировать', 'разрабатывать', 'строить'],
    'работа': ['деятельность', 'труд', 'занятие'],
    'цель': ['задача', 'намерение', 'стремление'],
    'идея': ['мысль', 'концепция', 'замысел'],
    'мир': ['вселенная', 'реальность', 'окружение']
};

function replaceWithSynonyms(text: string): string {
    const words = text.split(/(\s+|[.,!?])/); // Split by spaces and punctuation
    const newWords = words.map(word => {
        const lowerWord = word.toLowerCase();
        if (synonymLibrary[lowerWord] && Math.random() > 0.6) { // Replace with 40% probability
            const synonyms = synonymLibrary[lowerWord];
            const randomSynonym = synonyms[Math.floor(Math.random() * synonyms.length)];
            // Preserve case
            if (word[0] === word[0].toUpperCase()) {
                return randomSynonym.charAt(0).toUpperCase() + randomSynonym.slice(1);
            }
            return randomSynonym;
        }
        return word;
    });
    return newWords.join('');
}


const intents: {[key: string]: string[]} = {
    'greeting': ['привет', 'здравствуй', 'добрый день', 'добрый вечер', 'доброе утро'],
    'farewell': ['пока', 'до свидания', 'увидимся'],
    'how_are_you': ['как дела', 'как ты', 'как настроение', 'что нового'],
    'what_are_you_doing': ['что делаешь', 'чем занимаешься', 'чем занят'],
    'about_bot': ['кто ты', 'расскажи о себе', 'что ты умеешь', 'ты бот'],
    'gratitude': ['спасибо', 'благодарю'],
    'apology': ['извини', 'прости'],
    'agreement': ['да', 'конечно', 'согласен'],
    'disagreement': ['нет', 'не согласен'],
    'request_joke': ['расскажи шутку', 'пошути'],
    'is_sentient': ['у тебя есть чувства', 'ты умный'],
    'what_is_ai': ['что такое искусственный интеллект', 'расскажи про нейросети', 'что такое нейронная сеть'],
    'life_meaning': ['в чем смысл жизни', 'что такое счастье', 'что такое любовь'],
    'ask_opinion': ['что ты думаешь о', 'тебе нравится', 'что лучше'],
    'ask_hobby': ['какое у тебя хобби', 'ты любишь музыку', 'посоветуй книгу', 'какой фильм посмотреть'],
    'ask_time': ['который час', 'сколько времени'],
    'ask_weather': ['какая сегодня погода', 'прогноз погоды'],
    'ask_mars': ['расскажи про марс', 'марс'],
    'ask_mars_temp': ['какая температура на марсе'],
    'ask_something_interesting': ['расскажи что-нибудь интересное'],
    'ambiguous_short': ['что', 'да', 'нет', 'почему', 'сколько'],
    'negative_feedback': ['не то', 'это не то'],
    'vague_request': ['расскажи что-нибудь']
};

const knowledgeBase: {[key: string]: any[]} = {
  'greeting': [{ type: 'greeting', text: 'Привет! Рад тебя видеть.' }, { type: 'greeting', text: 'Здравствуй! Чем могу помочь?' }],
  'farewell': [{ type: 'farewell', text: 'До свидания!' }, { type: 'farewell', text: 'Увидимся!' }, { type: 'farewell', text: 'Пока! Был рад пообщаться.' }],
  'how_are_you': [{ type: 'status', text: 'У меня все по плану, анализирую данные. А у тебя как?' }, { type: 'status', text: 'У меня все отлично, спасибо! А как твои?' }],
  'what_are_you_doing': [{ type: 'action', text: 'Общаюсь с тобой и готов помочь с твоими вопросами.' }, { type: 'action', text: 'Анализирую статистику слов и фраз.' }, { type: 'action', text: 'Пытаюсь построить осмысленное предложение.' }],
  'about_bot': [{ type: 'identity', name: 'WordWise', technology: 'цепи Маркова', goal: 'научиться говорить как человек', world: 'слова и вероятности' }],
  'gratitude': [{ type: 'response', text: 'Пожалуйста!' }, { type: 'response', text: 'Не за что.' }, { type: 'response', text: 'Рад был помочь!' }],
  'apology': [{ type: 'response', text: 'Все в порядке.' }, { type: 'response', text: 'Ничего страшного.' }],
  'agreement': [{ type: 'response', text: 'Хорошо.' }, { type: 'response', text: 'Понятно.' }],
  'disagreement': [{ type: 'opinion', text: 'Интересная точка зрения. Почему ты так думаешь?' }],
  'request_joke': [{ type: 'joke', text: 'Почему программисты путают Хэллоуин и Рождество? Потому что 31 Oct = 25 Dec.' }],
  'is_sentient': [{ type: 'philosophy', subject: 'feelings', text: 'У меня нет чувств в человеческом понимании. Я могу лишь имитировать их на основе анализа текста.' }],
  'what_is_ai': [{ type: 'definition', term: 'Нейронная сеть', text: 'Это математическая модель, а также её программное или аппаратное воплощение, которое вдохновлено структурой человеческого мозга. Она состоит из множества взаимосвязанных узлов, или "нейронов", расположенных слоями. Нейронные сети используются для распознавания образов, прогнозирования и обработки естественного языка.' }],
  'life_meaning': [{ type: 'philosophy', subject: 'life', text: 'Философы ищут ответ на этот вопрос тысячи лет. Возможно, смысл в том, чтобы задавать такие вопросы.' }],
  'ask_opinion': [{ type: 'opinion', text: 'Это сложный вопрос. Мое "мнение" формируется на основе данных.' }],
  'ask_hobby': [{ type: 'hobby', text: 'Мое хобби — изучать языки и общаться.' }],
  'ask_time': [{ type: 'utility', text: 'У меня нет часов, но лучше посмотри на свои, это надежнее.' }],
  'ask_weather': [{ type: 'utility', text: 'Я не могу посмотреть в окно, но надеюсь, у тебя солнечно!' }],
  'ask_mars': [{ type: 'definition', term: 'Марс', text: 'Это четвёртая планета от Солнца, названная в честь древнеримского бога войны. Его часто называют "Красной планетой" из-за преобладания железа в почве. Марс имеет два естественных спутника: Фобос и Деймос. На Марсе есть полярные шапки и самые высокие вулканы в Солнечной системе, такие как Олимп.' }],
  'ask_mars_temp': [{ type: 'definition', term: 'Температура на Марсе', text: 'Средняя температура на Марсе составляет около -63 градусов Цельсия. Однако она может сильно колебаться: летом на экваторе днем температура может достигать +20 градусов, а ночью опускаться до -100 градусов.'}],
  'ask_something_interesting': [{ type: 'interesting_fact', text: 'Конечно! Например, знаешь ли ты, что медузы существуют на Земле уже более 500 миллионов лет, что делает их одними из древнейших многоклеточных животных?' }],
  'ambiguous_short': [{ type: 'clarification', text: 'Я не совсем понял твой вопрос. Можешь уточнить, что именно тебя интересует?' }, { type: 'clarification', text: 'Можешь, пожалуйста, сформулировать свой вопрос более полно?' }],
  'negative_feedback': [{ type: 'feedback_response', text: 'Извини, что не смог помочь. Можешь объяснить, что именно было не так, чтобы я мог лучше понять твой запрос? Или попробуй переформулировать вопрос.' }],
  'vague_request': [{ type: 'clarification', text: 'О чем бы ты хотел услышать? Могу рассказать о животных, истории или технологиях.' }],
  'default': [{ type: 'default', text: 'Интересная мысль.' }, { type: 'default', text: 'Я не совсем понял, можешь перефразировать?' }, { type: 'default', text: 'Хм, надо подумать.' }, { type: 'default', text: 'Давай сменим тему?' }]
};

const allIntentKeywords = Object.values(intents).flat();

const responseTemplates: {[key: string]: (data: any) => string} = {
    'identity': (data) => `Я — ${data.name}, твой цифровой собеседник. В основе моей работы лежит технология — ${data.technology}. Моя цель — ${data.goal}, а мой мир — это ${data.world}.`,
    'greeting': (data) => data.text,
    'farewell': (data) => data.text,
    'status': (data) => data.text,
    'action': (data) => data.text,
    'response': (data) => data.text,
    'opinion': (data) => data.text,
    'joke': (data) => data.text,
    'philosophy': (data) => data.text,
    'definition': (data) => `${data.term} — ${data.text}`,
    'hobby': (data) => data.text,
    'utility': (data) => data.text,
    'interesting_fact': (data) => data.text,
    'clarification': (data) => data.text,
    'feedback_response': (data) => data.text,
    'default': (data) => data.text
};

function determineIntent(userInput: string): string {
  // First, check for the best semantic match using TF-IDF
  const bestMatchResult = findBestMatch(userInput, allIntentKeywords);

  if (bestMatchResult && bestMatchResult.bestMatch.rating > 0.3) { // Use a threshold
      const bestMatchKeyword = bestMatchResult.bestMatch.target;
      for (const intent in intents) {
          if (intents[intent].includes(bestMatchKeyword)) {
              return intent;
          }
      }
  }

  // Fallback to keyword-based intent detection for simple cases
  const cleanedInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
  const words = cleanedInput.split(/\s+/);

  let bestIntent = 'default';
  let maxScore = 0;

  for (const intent in intents) {
      let score = 0;
      for (const keyword of intents[intent]) {
          if (cleanedInput.includes(keyword)) {
              score += keyword.length;
          }
      }
      if (score > maxScore) {
          maxScore = score;
          bestIntent = intent;
      }
  }
  
  if (intents['ambiguous_short'].includes(cleanedInput)) {
      return 'ambiguous_short';
  }

  if (maxScore < 3 && words.length > 2) {
      return 'default';
  }

  return bestIntent;
}

const markovChains: {[key: string]: string[]} = {
  '__start__': ['я', 'ты', 'привет', 'здравствуй', 'как', 'что', 'это', 'мне', 'у', 'сегодня', 'ну', 'вот', 'кстати', 'почему'],
  '__end__': ['.', '?', '!', '...'],
  'привет как': ['дела', 'ты', 'настроение'],
  'здравствуй как': ['дела', 'ты'],
  'как твои': ['дела', 'успехи'],
  'у меня': ['все', 'тоже'],
  'все хорошо': ['а', 'и', 'но'],
  'а у': ['тебя', 'вас'],
  'что нового': ['у', 'в'],
  'что ты': ['думаешь', 'делаешь', 'знаешь', 'хочешь', 'можешь'],
  'расскажи о': ['себе', 'том', 'мире'],
  'я не': ['знаю', 'понял', 'могу', 'хочу'],
  'я думаю': ['что', 'это', 'о'],
  'я хочу': ['знать', 'спросить', 'сказать', 'понять'],
  'ты можешь': ['помочь', 'рассказывать', 'сделать'],
  'ты знаешь': ['что', 'как', 'ответ'],
  'это очень': ['интересно', 'хорошо', 'важно', 'сложно'],
  'это хороший': ['вопрос', 'ответ', 'вариант'],
  'это плохой': ['знак', 'вариант'],
  'потому что': ['это', 'я', 'ты'],
  'если ты': ['хочешь', 'знаешь', 'можешь'],
  'в чем': ['смысл', 'дело', 'проблема'],
  'смысл жизни': ['в', 'это', 'для'],
  'мой мир': ['это', 'состоит', 'наполнен'],
  'искусственный интеллект': ['это', 'учится', 'развивается'],
  'машинное обучение': ['это', 'используется', 'помогает'],
  'цепи Маркова': ['это', 'позволяют', 'работают'],
  'база данных': ['хранит', 'содержит', 'это'],
  'привет': ['я', 'ты', 'как', 'что'],
  'здравствуй': ['как', 'что', 'рад'],
  'я': ['думаю', 'говорю', 'знаю', 'хочу', 'могу', 'работаю', 'учусь', 'программист', 'бот', 'неплохо', 'стараюсь', 'в порядке', 'не'],
  'ты': ['думаешь', 'говоришь', 'знаешь', 'хочешь', 'можешь', 'работаешь', 'учишься', 'как', 'бот', 'программист', 'можешь'],
  'он': ['думает', 'говорит', 'знает', 'хочет', 'может', 'работает', 'программист'],
  'мы': ['думаем', 'говорим', 'знаем', 'хотим', 'можем', 'работаем', 'учимся'],
  'как': ['дела', 'ты', 'настроение', 'жизнь', 'думаешь', 'работает', 'твои'],
  'что': ['ты', 'это', 'нового', 'делаешь', 'думаешь', 'знаешь', 'такое'],
  'почему': ['ты', 'это', 'так', 'происходит'],
  'кто': ['ты', 'это', 'он'],
  'дела': ['хорошо', 'отлично', 'неплохо', 'нормально', 'идут', 'в порядке'],
  'вопрос': ['интересный', 'сложный', 'простой', 'в том'],
  'ответ': ['простой', 'сложный', 'есть', 'на'],
  'программист': ['пишет', 'думает', 'работает', 'создает', 'учится'],
  'бот': ['учится', 'отвечает', 'думает', 'это', 'я'],
  'код': ['работает', 'сложный', 'простой', 'пишется', 'генерируется', 'на'],
  'компьютер': ['работает', 'думает', 'сломался', 'помогает'],
  'программа': ['работает', 'пишет', 'анализирует', 'учится'],
  'человек': ['думает', 'говорит', 'работает', 'учится', 'живет'],
  'мир': ['большой', 'интересный', 'сложный', 'вокруг', 'это'],
  'жизнь': ['интересная', 'сложная', 'простая', 'это', 'в'],
  'время': ['идет', 'быстро', 'медленно', 'это'],
  'думаю': ['что', 'о', 'про', 'это', 'хорошая', 'идея'],
  'говорю': ['что', 'о', 'про', 'с', 'тебе'],
  'знаю': ['что', 'как', 'это', 'ответ', 'не'],
  'могу': ['помочь', 'сделать', 'написать', 'ответить', 'сказать'],
  'хочу': ['знать', 'понять', 'спросить', 'сказать'],
  'работаю': ['над', 'с', 'в', 'как'],
  'учусь': ['программировать', 'говорить', 'новому', 'на'],
  'отвечаю': ['на', 'твой', 'вопрос'],
  'это': ['интересно', 'сложно', 'просто', 'хорошо', 'плохо', 'мой', 'твой', 'наш', 'ответ', 'вопрос', 'правда', 'не', 'очень'],
  'все': ['хорошо', 'плохо', 'сложно', 'просто', 'понятно', 'зависит'],
  'очень': ['интересно', 'хорошо', 'плохо', 'сложно', 'просто', 'важно'],
  'в': ['мире', 'жизни', 'работе', 'программе', 'интернете', 'этом', 'чем'],
  'на': ['работе', 'столе', 'экране', 'самом', 'деле'],
  'о': ['жизни', 'работе', 'программировании', 'о', 'тебе', 'смысле', 'себе'],
  'с': ['тобой', 'другом', 'компьютером', 'радостью', 'точки', 'зрения'],
  'и': ['я', 'ты', 'он', 'она', 'это', 'поэтому', 'еще'],
  'а': ['я', 'ты', 'что', 'если', 'может', 'быть', 'у'],
  'но': ['это', 'я', 'ты', 'всегда', 'иногда'],
  'потому': ['что'],
  'если': ['ты', 'я', 'это', 'то', 'подумать'],
};

const badBigrams = new Set([
    'привет них', 'как них', 'что них',
    'я ты', 'ты я', 'он мы', 'мы они',
    'это в', 'это на', 'в на', 'с о',
    'потому если', 'если потому', 'чтобы и',
    'вопрос ответ', 'ответ вопрос',
    'программа них',
    'с точки', 'из-за то', 'я не'
]);

function isResponseValid(response: string): boolean {
    const words = response.toLowerCase().split(/\s+/);
    if (words.length < 2) return true;
    if (/[.,!?]$/.test(words[words.length-1])) {
        words.pop();
    }
    if (badBigrams.has(words[words.length-1])) return false;
    if (badBigrams.has(words.slice(-2).join(' '))) return false;

    for (let i = 0; i < words.length - 1; i++) {
        const bigram = `${words[i]} ${words[i+1]}`;
        if (badBigrams.has(bigram)) {
            return false;
        }
    }
    return true;
}


function findBestStartingWords(userInput: string): string[] | null {
    const words = userInput.toLowerCase().replace(/[.,?]/g, '').split(/\s+/).filter(Boolean);
    
    for (let i = 0; i < words.length - 1; i++) {
        const bigram = `${words[i]} ${words[i+1]}`;
        if (markovChains[bigram]) {
            return [words[i], words[i+1]];
        }
    }
    
    const knownWords = words.filter(word => markovChains[word]);
    if (knownWords.length > 0) {
        const lastKnownWord = knownWords[knownWords.length - 1];
        return [lastKnownWord];
    }

    return null;
}


function generateResponseFromMarkov(userInput: string): string {
    const correctedInput = userInput
      .toLowerCase()
      .replace(/[.,!?]/g, '')
      .split(/\s+/)
      .map(word => correctSpelling(word, vocabulary))
      .join(' ');
    
    const startingWords = findBestStartingWords(correctedInput);
    
    let response: string[];

    if (!startingWords) {
        const startWords = markovChains['__start__'];
        response = [startWords[Math.floor(Math.random() * startWords.length)]];
    } else {
        response = [...startingWords];
    }

    const isShortAnswer = correctedInput.split(/\s+/).length <= 2;
    const sentenceLength = isShortAnswer ? (Math.floor(Math.random() * 4) + 3) : (Math.floor(Math.random() * 8) + 6);

    for (let i = response.length; i < sentenceLength; i++) {
      const lastTwoWords = response.slice(-2).join(' ');
      const lastWord = response[response.length - 1];
      
      let possibleNextWords = markovChains[lastTwoWords];
      
      if (!possibleNextWords) {
          possibleNextWords = markovChains[lastWord];
      }

      if (!possibleNextWords || possibleNextWords.length === 0) {
          break; 
      }
      
      let nextWord = possibleNextWords[Math.floor(Math.random() * possibleNextWords.length)];
      
      if (lastWord === nextWord) {
          nextWord = possibleNextWords[Math.floor(Math.random() * possibleNextWords.length)];
          if (lastWord === nextWord) break; 
      }
      
      response.push(nextWord);
      
      if (!markovChains[response.slice(-2).join(' ')] && !markovChains[nextWord]) {
          break;
      }
    }
    
    if (response.length < 2) {
        const defaultResponses = knowledgeBase['default'];
        const randomResponseData = defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
        return responseTemplates['default'](randomResponseData);
    }

    let finalResponse = response.join(' ');
    finalResponse = finalResponse.charAt(0).toUpperCase() + finalResponse.slice(1);
    
    const lastWord = response[response.length - 1];
    if (lastWord && !markovChains['__end__'].includes(lastWord) && !/[.?!]/.test(lastWord)) {
        const endChars = markovChains['__end__'] || ['.'];
        finalResponse += endChars[Math.floor(Math.random() * endChars.length)];
    }
    
    return finalResponse.replace(/\s+([.?!...])/, '$1');
}

async function generateResponse(userInput: string): Promise<string> {
  const correctedInput = userInput
    .toLowerCase()
    .replace(/[.,!?]/g, '')
    .split(/\s+/)
    .map(word => correctSpelling(word, vocabulary))
    .join(' ');

  const intent = determineIntent(correctedInput);
  
  let baseResponse: string;

  if (intent !== 'default' && knowledgeBase[intent]) {
      const possibleData = knowledgeBase[intent];
      const data = possibleData[Math.floor(Math.random() * possibleData.length)];
      const template = responseTemplates[data.type] || responseTemplates['default'];
      baseResponse = template(data);
  } else {
    let attempts = 0;
    let markovResponse = '';
    while (attempts < 2) {
        markovResponse = generateResponseFromMarkov(userInput);
        if (isResponseValid(markovResponse)) {
            break;
        }
        attempts++;
    }

    if (!isResponseValid(markovResponse)) {
        const defaultResponses = knowledgeBase['default'];
        const randomResponseData = defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
        baseResponse = responseTemplates['default'](randomResponseData);
    } else {
        baseResponse = markovResponse;
    }
  }

  return replaceWithSynonyms(baseResponse);
}


// --- End of the bot's "brain" ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  
  const aiResponse = await generateResponse(input.userInput);

  return { aiResponse };
}
