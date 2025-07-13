'use server';

/**
 * @fileOverview This file defines a flow for generating contextually relevant responses
 * using a Markov chain model with an expanded vocabulary and phrase recognition.
 *
 * - contextualResponse - A function that takes user input and returns a contextually relevant response.
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

const vocabulary: string[] = [
    // Greetings and Farewells
    'привет', 'здравствуй', 'добрый день', 'добрый вечер', 'пока', 'до свидания', 'увидимся',
    // Common questions and phrases
    'как дела', 'что нового', 'как ты', 'что делаешь', 'чем занимаешься', 'расскажи о себе', 'кто ты',
    // Common answers
    'хорошо', 'отлично', 'неплохо', 'так себе', 'замечательно', 'нормально', 'я в порядке', 'все хорошо',
    // Feelings and emotions
    'радость', 'грусть', 'удивление', 'скука', 'интерес', 'счастье', 'злость', 'спокойствие',
    // People and roles
    'я', 'ты', 'он', 'она', 'мы', 'вы', 'они', 'человек', 'друг', 'программист', 'собеседник', 'бот',
    // Actions
    'думать', 'говорить', 'писать', 'читать', 'смотреть', 'слушать', 'работать', 'учиться', 'помогать', 'знать', 'любить', 'хотеть', 'мочь',
    // Objects and concepts
    'слово', 'фраза', 'предложение', 'текст', 'код', 'программа', 'идея', 'мысль', 'вопрос', 'ответ', 'мир', 'жизнь', 'время', 'работа', 'компьютер',
    // Adjectives
    'интересный', 'скучный', 'новый', 'старый', 'хороший', 'плохой', 'большой', 'маленький', 'умный', 'простой', 'сложный',
    // Adverbs
    'сегодня', 'завтра', 'вчера', 'всегда', 'иногда', 'никогда', 'здесь', 'там', 'очень', 'немного', 'быстро', 'медленно',
    // Conjunctions and prepositions
    'и', 'а', 'но', 'или', 'потому что', 'если', 'что', 'чтобы', 'в', 'на', 'о', 'про', 'с', 'к', 'по', 'из',
    // Places
    'дом', 'город', 'страна', 'интернет',
    // Tech
    'алгоритм', 'данные', 'сеть', 'база данных', 'интерфейс', 'разработка', 'тестирование', 'облако',
    // Philosophy
    'смысл', 'сознание', 'бытие', 'знание', 'реальность',
    // Fillers
    'ну', 'эм', 'вот', 'кстати', 'знаешь',
    // More verbs
    'создавать', 'улучшать', 'анализировать', 'строить', 'генерировать', 'отвечать',
    // More nouns
    'язык', 'модель', 'вероятность', 'статистика', 'контекст', 'диалог', 'цель', 'задача',
    // ... up to 1000 words
];

const cannedResponses: {[key: string]: string[]} = {
  'как дела': ['У меня все отлично, спасибо! А у тебя?', 'Все хорошо, работаю над собой.', 'Нормально, учу новые слова.'],
  'кто ты': ['Я — WordWise, простой чат-бот.', 'Я — программа, которая учится говорить.', 'Я твой собеседник.'],
  'что ты умеешь': ['Я умею говорить, используя слова, которые знаю.', 'Я могу построить предложение. Иногда получается смешно.', 'Я учусь поддерживать диалог.'],
  'расскажи о себе': ['Я живу в интернете. Моя цель — научиться говорить как человек. Мой словарный запас ограничен, но я стараюсь.'],
  'привет': ['Привет!', 'Здравствуй!', 'Добрый день!'],
  'пока': ['До свидания!', 'Увидимся!', 'Пока!'],
};


const markovChains: {[key: string]: string[]} = {
  'привет': ['я', 'ты', 'как'],
  'здравствуй': ['как', 'что'],
  'я': ['думаю', 'говорю', 'люблю', 'хочу', 'работаю', 'программист', 'бот', 'неплохо'],
  'ты': ['думаешь', 'говоришь', 'знаешь', 'программист', 'бот', 'как'],
  'как': ['дела', 'ты', 'жизнь', 'программа', 'работа'],
  'дела': ['хорошо', 'отлично', 'неплохо', 'идут'],
  'что': ['делаешь', 'нового', 'ты', 'это', 'такое'],
  'мой': ['мир', 'дом', 'компьютер', 'код'],
  'твой': ['мир', 'дом', 'код', 'ответ'],
  'мне': ['интересно', 'кажется', 'нужно'],
  'тебе': ['интересно', 'кажется', 'нужно'],
  'он': ['программист', 'думает', 'работает'],
  'она': ['программист', 'думает', 'работает'],
  'мы': ['говорим', 'работаем', 'учимся'],
  'вы': ['говорите', 'работаете', 'учитесь'],
  'они': ['говорят', 'работают', 'учатся'],
  'это': ['интересно', 'сложно', 'просто', 'хорошо', 'плохо', 'мой', 'твой', 'наш'],
  'все': ['хорошо', 'плохо', 'сложно', 'просто'],
  'у': ['меня', 'тебя', 'него', 'нас', 'вас', 'них'],
  'меня': ['зовут', 'есть', 'нет'],
  'мой': ['друг', 'код', 'проект'],
  'компьютер': ['работает', 'думает', 'сломался'],
  'программа': ['работает', 'пишет', 'анализирует'],
  'код': ['работает', 'сложный', 'простой', 'пишется'],
  'человек': ['думает', 'говорит', 'работает'],
  'друг': ['помогает', 'говорит', 'знает'],
  'мир': ['большой', 'интересный', 'сложный'],
  'жизнь': ['интересная', 'сложная', 'простая'],
  'время': ['идет', 'быстро', 'медленно'],
  'работа': ['интересная', 'сложная', 'простая'],
  'думать': ['о', 'про', 'что', 'как'],
  'говорить': ['о', 'про', 'что', 'с'],
  'писать': ['код', 'текст', 'программу'],
  'читать': ['книгу', 'текст', 'код'],
  'смотреть': ['на', 'в', 'кино'],
  'любить': ['программировать', 'читать', 'ты'],
  'хотеть': ['знать', 'уметь', 'спать'],
  'мочь': ['помочь', 'сделать', 'написать'],
  'в': ['мире', 'жизни', 'работе', 'программе', 'интернете'],
  'на': ['работе', 'столе', 'экране'],
  'о': ['жизни', 'работе', 'программировании'],
  'про': ['жизнь', 'работу', 'программирование'],
  'с': ['тобой', 'другом', 'компьютером'],
  'и': ['я', 'ты', 'он', 'она', 'это'],
  'а': ['я', 'ты', 'что'],
  'но': ['это', 'я', 'ты'],
  'потому что': ['это', 'я', 'ты'],
  'если': ['ты', 'я', 'это'],
  '__start__': ['я', 'ты', 'привет', 'здравствуй', 'как', 'что', 'это', 'мне', 'у', 'сегодня'],
  '__end__': ['.', '?', '!']
};


function findBestStartingWord(userInput: string): string {
    const words = userInput.toLowerCase().split(/\s+/);
    const knownWords = words.filter(word => markovChains[word]);
    if (knownWords.length > 0) {
        return knownWords[Math.floor(Math.random() * knownWords.length)];
    }
    return '__start__';
}

function generateResponse(userInput: string): string {
  // Check for canned responses first for whole phrases
  for (const phrase in cannedResponses) {
    if (userInput.toLowerCase().includes(phrase)) {
      const possibleResponses = cannedResponses[phrase];
      return possibleResponses[Math.floor(Math.random() * possibleResponses.length)];
    }
  }

  let currentWord = findBestStartingWord(userInput);
  if (currentWord === '__start__') {
      const startWords = markovChains['__start__'];
      currentWord = startWords[Math.floor(Math.random() * startWords.length)];
  }

  let response = [currentWord];
  const sentenceLength = Math.floor(Math.random() * 8) + 4; // 4 to 11 words

  for (let i = 0; i < sentenceLength; i++) {
    const possibleNextWords = markovChains[currentWord] || vocabulary;
    const nextWord = possibleNextWords[Math.floor(Math.random() * possibleNextWords.length)];
    
    // Avoid immediate repetition
    if (response[response.length - 1] !== nextWord) {
        response.push(nextWord);
    }
    
    currentWord = nextWord;
    if (!markovChains[currentWord]) {
        // If we hit a word with no chain, break to avoid getting stuck
        break;
    }
  }

  let finalResponse = response.join(' ');
  finalResponse = finalResponse.charAt(0).toUpperCase() + finalResponse.slice(1);
  
  const lastChar = finalResponse[finalResponse.length - 1];
  if (lastChar !== '.' && lastChar !== '?' && lastChar !== '!') {
      const endChars = markovChains['__end__'] || ['.'];
      finalResponse += endChars[Math.floor(Math.random() * endChars.length)];
  }
  
  return finalResponse;
}

// --- End of the bot's "brain" ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  
  const aiResponse = generateResponse(input.userInput);

  return { aiResponse };
}
