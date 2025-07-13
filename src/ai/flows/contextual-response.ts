'use server';

/**
 * @fileOverview This file defines a flow for generating contextually relevant responses.
 *
 * - contextualResponse - A function that takes user input and returns a contextually relevant response.
 * - ContextualResponseInput - The input type for the contextualResponse function.
 * - ContextualResponseOutput - The return type for the contextualResponse function.
 */

import { z } from 'zod';

const ContextualResponseInputSchema = z.object({
  userInput: z
    .string()
    .describe('The user input to which the AI should respond.'),
});
export type ContextualResponseInput = z.infer<typeof ContextualResponseInputSchema>;

const ContextualResponseOutputSchema = z.object({
  aiResponse: z.string().describe('The AI response based on the user input.'),
});
export type ContextualResponseOutput = z.infer<typeof ContextualResponseOutputSchema>;

// --- Модель на основе цепей Маркова ---

const transitions: Record<string, string[]> = {
  __start: ['Привет', 'Здравствуй', 'Как', 'Что', 'Расскажи', 'Я', 'Мне'],
  привет: ['! Как', '!', 'здравствуй', '.', 'чем', 'могу', 'помочь', '?'],
  здравствуй: ['!', 'как', 'почему', '.', 'рад', 'тебя', 'видеть', '.'],
  как: ['дела', 'погода', 'ты', 'настроение', 'это', 'возможно', '?'],
  дела: ['?', 'у', 'меня', 'хорошо', 'плохо', 'отлично', ',', 'а', 'у', 'тебя', '?'],
  что: ['нового', 'делаешь', 'интересного', 'ты', 'думаешь', 'про', 'это', '?'],
  нового: ['?', 'у', 'меня', 'ничего', 'особенного', '.', 'А', 'у', 'тебя', '?'],
  хорошо: ['.', 'а', 'у', 'тебя', '?', 'очень', 'рад', 'это', 'слышать', '.'],
  плохо: ['.', 'почему', '?', 'что', 'случилось', '?'],
  почему: ['?', 'ты', 'так', 'думаешь', '?', 'это', 'интересный', 'вопрос', '.'],
  сегодня: ['хорошая', 'плохая', 'погода', '.', 'отличный', 'день', '.'],
  завтра: ['будет', 'новый', 'день', '.', 'посмотрим', ',', 'что', 'будет', '.'],
  погода: ['хорошая', 'плохая', 'солнечная', 'дождливая', 'ветреная', '.'],
  я: ['думаю', 'знаю', 'хочу', 'работаю', 'отдыхаю', 'люблю', 'программировать', '.'],
  ты: ['думаешь', 'знаешь', 'хочешь', 'работаешь', 'можешь', 'любишь', '?'],
  думаю: ['что', 'все', 'будет', 'хорошо', '.', 'об', 'этом', 'нужно', 'подумать', '.'],
  знаю: ['что', 'это', 'интересно', '.', 'не', 'очень', 'много', ',', 'но', 'учусь', '.'],
  хочу: ['спать', 'есть', 'узнать', 'больше', '.', 'поговорить', 'с', 'тобой', '.'],
  работаю: ['над', 'собой', '.', 'не', 'покладая', 'рук', '.'],
  отдыхаю: ['когда', 'нет', 'задач', '.', 'редко', ',', 'но', 'метко', '.'],
  спасибо: ['.', 'пожалуйста', '!', 'не', 'за', 'что', '!'],
  пожалуйста: ['.', 'не', 'за', 'что', '.', 'обращайся', 'еще', '.'],
  расскажи: ['мне', 'о', 'себе', 'что-нибудь', 'интересное', '.'],
  мне: ['интересно', 'скучно', 'кажется', ',', 'что', '.'],
  о: ['тебе', 'работе', 'погоде', 'любви', 'программировании', '.'],
  тебе: ['?', 'интересно', 'скучно', 'нравится', 'это', '?'],
  пока: ['.', 'до', 'свидания', '!', 'был', 'рад', 'пообщаться', '.'],
  интересно: ['!.', 'почему', 'ты', 'так', 'думаешь', '?'],
  скучно: ['.', 'может', 'поговорим', 'о', 'чем-то', 'другом', '?'],
  рад: ['тебя', 'видеть', '.', 'слышать', 'это', '.'],
  могу: ['помочь', '?', 'попробовать', 'ответить', '.'],
  у: ['меня', 'тебя', 'нас', 'все', 'хорошо', '.'],
  меня: ['зовут', 'WordWise', '.', 'все', 'хорошо', '.'],
  тебя: ['?', 'как', 'зовут', '?'],
  это: ['интересно', 'странно', 'хорошо', 'плохо', 'невероятно', '.'],
  все: ['будет', 'хорошо', '.', 'отлично', '!', 'сложно', '.'],
  люблю: ['программировать', 'общаться', 'узнавать', 'новое', '.'],
  программировать: ['это', 'мое', 'хобби', '.', 'на', 'JavaScript', '.'],
  вопрос: ['.', 'требует', 'размышлений', '.'],
  ответ: ['прост', '.', 'не', 'так', 'очевиден', '.'],
};

const keywords: Record<string, string[]> = {
  привет: ['привет', 'здравствуй', 'добрый день', 'хелло'],
  пока: ['пока', 'до свидания', 'удачи', 'до встречи'],
  погода: ['погода', 'солнце', 'дождь', 'ветер', 'прогноз'],
  дела: ['дела', 'как ты', 'как жизнь', 'что нового'],
  спасибо: ['спасибо', 'благодарю'],
  'расскажи о себе': ['кто ты', 'о себе', 'расскажи', 'что ты умеешь'],
  'как настроение': ['настроение', 'духе'],
  'что делаешь': ['делаешь', 'занят'],
};

function getRandomElement<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function generateResponse(userInput: string): string {
  const lowerCaseInput = userInput.toLowerCase();
  
  let startWord: string | null = null;

  // 1. Ищем ключевые слова в сообщении пользователя, чтобы выбрать тему ответа
  for (const start of Object.keys(keywords)) {
    if (keywords[start].some(kw => lowerCaseInput.includes(kw))) {
       const possibleStarts: Record<string, string[]> = {
          'привет': ['Привет!', 'Здравствуй!', 'Добрый день!'],
          'пока': ['Пока!', 'До встречи!', 'Удачи!'],
          'погода': ['Сегодня', 'Думаю,', 'Завтра', 'Погода сегодня'],
          'дела': ['У меня все отлично,', 'Дела идут своим чередом.', 'Всё хорошо, а у тебя как?'],
          'спасибо': ['Пожалуйста!', 'Не за что.', 'Всегда рад помочь.'],
          'расскажи о себе': ['Я — простой бот.', 'Что именно тебя интересует?', 'Я — чат-бот, основанный на цепях Маркова.'],
          'как настроение': ['Настроение отличное!', 'Боевое!', 'Всё в порядке.'],
          'что делаешь': ['Общаюсь с тобой!', 'Пытаюсь сгенерировать осмысленный ответ.', 'Работаю.'],
       };
       startWord = getRandomElement(possibleStarts[start] || transitions.__start);
       break;
    }
  }

  // 2. Если ключевых слов не найдено, начинаем со случайного слова
  if (!startWord) {
    startWord = getRandomElement(transitions.__start);
  }

  // Особый случай для "привет" с 80% вероятностью
  if (keywords['привет'].some(kw => lowerCaseInput.includes(kw)) && Math.random() < 0.8) {
      startWord = getRandomElement(['Привет!', 'Здравствуй!']);
  }


  let response = [startWord];
  let currentWord = startWord.toLowerCase().replace(/[.!?]/g, '').split(' ').pop() || '';
  const responseLength = Math.floor(Math.random() * 8) + 5; // Длина ответа от 5 до 12 слов

  // 3. Генерируем продолжение ответа на основе цепей Маркова
  for (let i = 0; i < responseLength; i++) {
    const possibleNextWords = transitions[currentWord] || transitions.__start.map(w => w.toLowerCase());
    let nextWord = getRandomElement(possibleNextWords);
    
    // Предотвращаем зацикливание и глупые повторы
    let attempts = 0;
    while ((nextWord === currentWord || response.includes(nextWord)) && attempts < 5) {
        nextWord = getRandomElement(possibleNextWords);
        attempts++;
    }
    
    if (['.', '?', '!'].includes(nextWord)) {
        if (response.length > 1) {
          response[response.length - 1] += nextWord;
        }
        break; 
    }
    
    response.push(nextWord);
    currentWord = nextWord.replace(/[.!?]/g, '');
    if (!transitions[currentWord]) {
      // Если для слова нет продолжения, прерываемся, чтобы не начать с начала
      break;
    }
  }
  
  // 4. Форматируем финальный ответ
  let finalResponse = response.join(' ');
  // Убираем пробелы перед знаками препинания
  finalResponse = finalResponse.replace(/\s+([,.!?])/g, '$1');
   // Убираем запятую в конце, если она там оказалась
  finalResponse = finalResponse.replace(/,$/, '.');
  // Добавляем точку в конце, если ее нет
  if (!/[.!?]$/.test(finalResponse)) {
      finalResponse += '.';
  }

  return finalResponse.charAt(0).toUpperCase() + finalResponse.slice(1);
}


export async function contextualResponse(input: ContextualResponseInput): Promise<ContextualResponseOutput> {
  // Имитация задержки ответа
  await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 500));
  
  const responseText = generateResponse(input.userInput);

  return {
    aiResponse: responseText,
  };
}
