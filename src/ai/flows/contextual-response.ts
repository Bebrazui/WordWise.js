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


// --- Простая вероятностная модель ---

const vocabulary: string[] = [
  'привет', 'здравствуй', 'как', 'дела', 'что', 'нового', 'хорошо', 'плохо', 'почему',
  'сегодня', 'завтра', 'погода', 'солнечно', 'дождливо', 'я', 'ты', 'думаю', 'знаю',
  'может', 'быть', 'всегда', 'иногда', 'никогда', 'спасибо', 'пожалуйста', 'извини',
  'да', 'нет', 'конечно', 'возможно', 'расскажи', 'мне', 'о', 'тебе', 'интересно',
  'скучно', 'работа', 'отдых', 'планы', 'вечер', 'утро', 'день', 'ночь', 'пока',
];

function generateResponse(userInput: string): string {
  const words = userInput.toLowerCase().split(/\s+/);
  const firstWord = words[0];
  
  let responseWords: string[] = [];

  // 1. Обработка слова "привет" с 80% вероятностью
  if (firstWord === 'привет') {
    if (Math.random() < 0.8) {
      responseWords.push('Привет!');
    } else {
      // В остальных 20% случаев выбираем другой ответ
      const alternativeGreetings = ['Здравствуй!', 'Добрый день!'];
      responseWords.push(alternativeGreetings[Math.floor(Math.random() * alternativeGreetings.length)]);
    }
  } else {
    // Для других слов начинаем ответ со случайного слова
     responseWords.push(vocabulary[Math.floor(Math.random() * vocabulary.length)]);
     responseWords[0] = responseWords[0].charAt(0).toUpperCase() + responseWords[0].slice(1);
  }

  // 2. Генерация остальной части предложения
  const responseLength = Math.floor(Math.random() * 5) + 3; // Длина ответа от 3 до 7 слов

  while (responseWords.length < responseLength) {
    const nextWord = vocabulary[Math.floor(Math.random() * vocabulary.length)];
    // Избегаем повторения слов подряд
    if (responseWords[responseWords.length - 1] !== nextWord) {
      responseWords.push(nextWord);
    }
  }

  return responseWords.join(' ') + '.';
}


export async function contextualResponse(input: ContextualResponseInput): Promise<ContextualResponseOutput> {
  // Имитация задержки ответа
  await new Promise(resolve => setTimeout(resolve, 500));
  
  const responseText = generateResponse(input.userInput);

  return {
    aiResponse: responseText,
  };
}
