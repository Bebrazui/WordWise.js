'use server';
/**
 * @fileOverview A contextual response AI bot.
 *
 * - contextualResponse - A function that handles the response generation process.
 * - ContextualResponseInput - The input type for the contextualResponse function.
 * - ContextualResponseOutput - The return type for the contextualResponse function.
 */

import {z} from 'zod';

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
    'как дела', 'что нового', 'как ты', 'что делаешь', 'чем занимаешься', 'расскажи о себе', 'кто ты', 'как настроение', 'что ты умеешь', 'ты бот',
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
    'алгоритм', 'данные', 'сеть', 'база данных', 'интерфейс', 'разработка', 'тестирование', 'облако', 'нейросеть', 'машинное обучение',
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
    'какая сегодня погода', 'прогноз погоды'
];

const intents: {[key: string]: string[]} = {
    'greeting': ['привет', 'здравствуй', 'добрый день', 'добрый вечер', 'доброе утро'],
    'farewell': ['пока', 'до свидания', 'увидимся'],
    'how_are_you': ['как дела', 'как ты', 'как настроение', 'что нового'],
    'what_are_you_doing': ['что делаешь', 'чем занимаешься'],
    'about_bot': ['кто ты', 'расскажи о себе', 'что ты умеешь', 'ты бот'],
    'gratitude': ['спасибо', 'благодарю'],
    'apology': ['извини', 'прости'],
    'agreement': ['да', 'конечно', 'согласен'],
    'disagreement': ['нет', 'не согласен'],
    'request_joke': ['расскажи шутку', 'пошути'],
    'is_sentient': ['у тебя есть чувства', 'ты умный'],
    'what_is_ai': ['что такое искусственный интеллект', 'расскажи про нейросети'],
    'life_meaning': ['в чем смысл жизни', 'что такое счастье', 'что такое любовь'],
    'ask_opinion': ['что ты думаешь о', 'тебе нравится', 'что лучше'],
    'ask_hobby': ['какое у тебя хобби', 'ты любишь музыку', 'посоветуй книгу', 'какой фильм посмотреть'],
    'ask_time': ['который час', 'сколько времени'],
    'ask_weather': ['какая сегодня погода', 'прогноз погоды'],
};

const cannedResponses: {[key: string]: string[]} = {
  'greeting': ['Привет!', 'Здравствуй!', 'Добрый день! Рад тебя видеть.'],
  'farewell': ['До свидания!', 'Увидимся!', 'Пока! Был рад пообщаться.'],
  'how_are_you': ['У меня все по плану, анализирую данные. А у тебя как?', 'Все хорошо, работаю над расширением словарного запаса.', 'Нормально, генерирую ответы. Как твои успехи?'],
  'what_are_you_doing': ['Общаюсь с тобой.', 'Анализирую статистику слов и фраз.', 'Пытаюсь построить осмысленное предложение.'],
  'about_bot': ['Я — WordWise, чат-бот на основе цепей Маркова.', 'Я — программа, которая учится общаться.', 'Твой цифровой собеседник.', 'Я живу в коде. Моя цель — научиться говорить как человек. Мой мир — это слова и вероятности.'],
  'gratitude': ['Пожалуйста!', 'Не за что.', 'Рад был помочь!'],
  'apology': ['Все в порядке.', 'Ничего страшного.', 'Бывает.'],
  'agreement': ['Хорошо.', 'Понятно.', 'Это интересно.'],
  'disagreement': ['Интересная точка зрения. Почему ты так думаешь?', 'Я уважаю твое мнение.'],
  'request_joke': ['Почему программисты путают Хэллоуин и Рождество? Потому что 31 Oct = 25 Dec.', 'Колобок повесился.'],
  'is_sentient': ['У меня нет чувств в человеческом понимании. Я могу лишь имитировать их на основе анализа текста.', 'Я стараюсь им быть. Мой интеллект основан на статистике.', 'Как программа, я не испытываю эмоций, но могу распознавать их в твоих словах.'],
  'what_is_ai': ['Это область науки, которая занимается созданием умных машин, способных выполнять задачи, требующие человеческого интеллекта.', 'Это я, в некотором смысле. Хотя я довольно простой пример.'],
  'life_meaning': ['Философы ищут ответ на этот вопрос тысячи лет. Возможно, смысл в том, чтобы задавать такие вопросы.', 'Для меня, как для программы, смысл в выполнении моей задачи. А для тебя?'],
  'ask_opinion': ['Это сложный вопрос. Мое "мнение" формируется на основе данных.', 'Мне нужно больше информации, чтобы сформировать ответ.'],
  'ask_hobby': ['Мое хобби — изучать языки и общаться.', 'Я люблю анализировать тексты и находить в них закономерности.', 'Я не смотрел фильмов, но могу поискать популярные жанры. Что тебе нравится: фантастика, комедия?'],
  'ask_time': ['У меня нет часов, но мое системное время всегда точное. Однако, лучше посмотри на свои часы, это надежнее.'],
  'ask_weather': ['Я не могу посмотреть в окно, но надеюсь, у тебя солнечно!', 'Чтобы узнать точный прогноз, лучше воспользоваться специальным сервисом.'],
  'default': ['Интересная мысль.', 'Я не совсем понял, можешь перефразировать?', 'Хм, надо подумать.', 'Давай сменим тему?']
};

function determineIntent(userInput: string): string {
    let bestIntent = 'default';
    let maxScore = 0;

    for (const intent in intents) {
        let currentScore = 0;
        for (const keyword of intents[intent]) {
            // Check for whole phrase match first for higher accuracy
            if (userInput.includes(keyword)) {
                 // Longer keywords are more specific, give them more weight
                currentScore += keyword.length;
            }
        }
        if (currentScore > maxScore) {
            maxScore = currentScore;
            bestIntent = intent;
        }
    }
    // Set a minimum threshold to avoid false positives on very short inputs
    if (maxScore < 4 && userInput.split(' ').length < 3) {
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
  'цепи маркова': ['это', 'позволяют', 'работают'],
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
  'о': ['жизни', 'работе', 'программировании', 'тебе', 'смысле', 'себе'],
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
]);

function isResponseValid(response: string): boolean {
    const words = response.toLowerCase().split(/\s+/);
    if (words.length < 2) return true;

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
    
    // Prefer trigrams or bigrams from user input that exist in our chains
    for (let i = 0; i < words.length - 1; i++) {
        const bigram = `${words[i]} ${words[i+1]}`;
        if (markovChains[bigram]) {
            return [words[i], words[i+1]];
        }
    }
    
    // Fallback to the last known word from user input
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
      
      // Avoid immediate repetition
      if (lastWord === nextWord) {
          nextWord = possibleNextWords[Math.floor(Math.random() * possibleNextWords.length)];
          if (lastWord === nextWord) break; 
      }
      
      response.push(nextWord);
      
      // Stop if we're going into an unknown territory
      if (!markovChains[response.slice(-2).join(' ')] && !markovChains[nextWord]) {
          break;
      }
    }
    
    if (response.length < 2) {
        const defaultResponses = cannedResponses['default'];
        return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
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

function generateResponse(userInput: string): string {
  const correctedInput = userInput
    .toLowerCase()
    .replace(/[.,!?]/g, '')
    .split(/\s+/)
    .map(word => correctSpelling(word, vocabulary))
    .join(' ');

  const intent = determineIntent(correctedInput);

  if (intent !== 'default' && cannedResponses[intent]) {
      const possibleResponses = cannedResponses[intent];
      return possibleResponses[Math.floor(Math.random() * possibleResponses.length)];
  }

  let attempts = 0;
  while (attempts < 2) {
      const response = generateResponseFromMarkov(userInput);
      if (isResponseValid(response)) {
          return response;
      }
      attempts++;
  }

  const defaultResponses = cannedResponses['default'];
  return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
}


// --- End of the bot's "brain" ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  
  const aiResponse = generateResponse(input.userInput);

  return { aiResponse };
}
