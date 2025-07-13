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
  __start: ['Привет', 'Здравствуй', 'Как', 'Что', 'Расскажи'],
  привет: ['! Как', '!', 'здравствуй', '.'],
  здравствуй: ['!', 'как', 'почему', '.'],
  как: ['дела', 'погода', 'ты', 'настроение', '?'],
  дела: ['?', 'у', 'меня', 'хорошо', 'плохо', '.'],
  что: ['нового', 'делаешь', 'интересного', 'ты', '?'],
  нового: ['?', 'у', 'меня', 'ничего', '.'],
  хорошо: ['.', 'а', 'у', 'тебя', '?'],
  плохо: ['.', 'почему', '?'],
  почему: ['?', 'ты', 'так', 'думаешь', '?'],
  сегодня: ['хорошая', 'плохая', 'погода', '.'],
  завтра: ['будет', 'новый', 'день', '.'],
  погода: ['хорошая', 'плохая', 'солнечная', 'дождливая', '.'],
  я: ['думаю', 'знаю', 'хочу', 'работаю', 'отдыхаю'],
  ты: ['думаешь', 'знаешь', 'хочешь', 'работаешь', '?'],
  думаю: ['что', 'все', 'будет', 'хорошо', '.'],
  знаю: ['что', 'это', 'интересно', '.'],
  спасибо: ['.', 'пожалуйста', '.'],
  пожалуйста: ['.', 'не', 'за', 'что', '.'],
  расскажи: ['мне', 'о', 'себе', 'что-нибудь', '.'],
  мне: ['интересно', 'скучно', '.'],
  о: ['тебе', 'работе', 'погоде', '.'],
  тебе: ['?', 'интересно', 'скучно', '.'],
  пока: ['.', 'до', 'свидания', '.'],
};

const keywords: Record<string, string[]> = {
  привет: ['привет', 'здравствуй'],
  пока: ['пока', 'до свидания'],
  погода: ['погода', 'солнце', 'дождь'],
  дела: ['дела', 'как ты'],
  спасибо: ['спасибо'],
  'расскажи о себе': ['кто ты', 'о себе', 'расскажи'],
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
          'привет': ['Привет!', 'Здравствуй!'],
          'пока': ['Пока!', 'До встречи!'],
          'погода': ['Сегодня', 'Думаю,', 'Завтра'],
          'дела': ['У меня все хорошо,', 'Дела идут', 'Как дела?'],
          'спасибо': ['Пожалуйста!', 'Не за что.'],
          'расскажи о себе': ['Я — простой бот.', 'Что именно тебя интересует?'],
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
  if (lowerCaseInput.startsWith('привет') && Math.random() < 0.8) {
      startWord = 'Привет!';
  }


  let response = [startWord];
  let currentWord = startWord.toLowerCase().replace(/[.!?]/g, '');
  const responseLength = Math.floor(Math.random() * 6) + 4; // Длина ответа от 4 до 9 слов

  // 3. Генерируем продолжение ответа на основе цепей Маркова
  for (let i = 0; i < responseLength; i++) {
    const possibleNextWords = transitions[currentWord] || transitions.__start.map(w => w.toLowerCase());
    let nextWord = getRandomElement(possibleNextWords);
    
    // Предотвращаем зацикливание и глупые повторы
    if (nextWord === currentWord || response.includes(nextWord)) {
        nextWord = getRandomElement(possibleNextWords);
    }
    
    if (['.', '?', '!'].includes(nextWord)) {
        response[response.length - 1] += nextWord;
        break; 
    }
    
    response.push(nextWord);
    currentWord = nextWord;
  }
  
  // 4. Форматируем финальный ответ
  let finalResponse = response.join(' ');
  // Убираем пробелы перед знаками препинания
  finalResponse = finalResponse.replace(/\s+([.!?])/g, '$1');
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
