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
  'как ты': ['Все хорошо, спасибо. А ты как?', 'Я в порядке. Спасибо, что спросил.'],
  'кто ты': ['Я — WordWise, простой чат-бот.', 'Я — программа, которая учится говорить.', 'Я твой собеседник.'],
  'что ты умеешь': ['Я умею говорить, используя слова, которые знаю.', 'Я могу построить предложение. Иногда получается смешно.', 'Я учусь поддерживать диалог.'],
  'что делаешь': ['Общаюсь с тобой.', 'Учусь новым словам.', 'Думаю о смысле жизни.'],
  'расскажи о себе': ['Я живу в интернете. Моя цель — научиться говорить как человек. Мой словарный запас ограничен, но я стараюсь.'],
  'привет': ['Привет!', 'Здравствуй!', 'Добрый день!'],
  'здравствуй': ['И тебе здравствуй!', 'Привет!'],
  'пока': ['До свидания!', 'Увидимся!', 'Пока!'],
  'да': ['Хорошо.', 'Понятно.', 'Это интересно.'],
  'нет': ['Почему?', 'Жаль.', 'Ясно.'],
  'спасибо': ['Пожалуйста!', 'Не за что.'],
};


const markovChains: {[key: string]: string[]} = {
  'привет': ['я', 'ты', 'как'],
  'здравствуй': ['как', 'что'],
  'я': ['думаю', 'говорю', 'люблю', 'хочу', 'работаю', 'программист', 'бот', 'неплохо', 'учусь', 'стараюсь'],
  'ты': ['думаешь', 'говоришь', 'знаешь', 'программист', 'бот', 'как', 'хочешь'],
  'как': ['дела', 'ты', 'жизнь', 'программа', 'работа', 'настроение'],
  'дела': ['хорошо', 'отлично', 'неплохо', 'идут'],
  'что': ['делаешь', 'нового', 'ты', 'это', 'такое', 'думаешь'],
  'мой': ['мир', 'дом', 'компьютер', 'код', 'ответ'],
  'твой': ['мир', 'дом', 'код', 'ответ', 'вопрос'],
  'мне': ['интересно', 'кажется', 'нужно'],
  'тебе': ['интересно', 'кажется', 'нужно'],
  'он': ['программист', 'думает', 'работает', 'говорит'],
  'она': ['программист', 'думает', 'работает', 'говорит'],
  'мы': ['говорим', 'работаем', 'учимся', 'думаем'],
  'вы': ['говорите', 'работаете', 'учитесь', 'думаете'],
  'они': ['говорят', 'работают', 'учатся', 'думают'],
  'это': ['интересно', 'сложно', 'просто', 'хорошо', 'плохо', 'мой', 'твой', 'наш', 'ответ', 'вопрос'],
  'все': ['хорошо', 'плохо', 'сложно', 'просто', 'понятно'],
  'у': ['меня', 'тебя', 'него', 'нас', 'вас', 'них'],
  'меня': ['зовут', 'есть', 'нет'],
  'мой': ['друг', 'код', 'проект', 'вопрос'],
  'компьютер': ['работает', 'думает', 'сломался', 'помогает'],
  'программа': ['работает', 'пишет', 'анализирует', 'учится'],
  'код': ['работает', 'сложный', 'простой', 'пишется', 'генерируется'],
  'человек': ['думает', 'говорит', 'работает', 'учится'],
  'друг': ['помогает', 'говорит', 'знает'],
  'мир': ['большой', 'интересный', 'сложный'],
  'жизнь': ['интересная', 'сложная', 'простая'],
  'время': ['идет', 'быстро', 'медленно'],
  'работа': ['интересная', 'сложная', 'простая'],
  'думать': ['о', 'про', 'что', 'как', 'очень', 'много'],
  'говорить': ['о', 'про', 'что', 'с', 'с тобой'],
  'писать': ['код', 'текст', 'программу', 'ответ'],
  'читать': ['книгу', 'текст', 'код'],
  'смотреть': ['на', 'в', 'кино'],
  'любить': ['программировать', 'читать', 'ты'],
  'хотеть': ['знать', 'уметь', 'спать', 'помочь'],
  'мочь': ['помочь', 'сделать', 'написать', 'ответить'],
  'в': ['мире', 'жизни', 'работе', 'программе', 'интернете'],
  'на': ['работе', 'столе', 'экране', 'самом деле'],
  'о': ['жизни', 'работе', 'программировании', 'тебе'],
  'про': ['жизнь', 'работу', 'программирование', 'это'],
  'с': ['тобой', 'другом', 'компьютером', 'радостью'],
  'и': ['я', 'ты', 'он', 'она', 'это', 'поэтому'],
  'а': ['я', 'ты', 'что', 'если'],
  'но': ['это', 'я', 'ты', 'всегда'],
  'потому что': ['это', 'я', 'ты', 'так', 'надо'],
  'если': ['ты', 'я', 'это', 'то'],
  '__start__': ['я', 'ты', 'привет', 'здравствуй', 'как', 'что', 'это', 'мне', 'у', 'сегодня', 'ну'],
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
  const lowerUserInput = userInput.toLowerCase().trim();

  // Check for canned responses first for whole phrases
  if (cannedResponses[lowerUserInput]) {
      const possibleResponses = cannedResponses[lowerUserInput];
      return possibleResponses[Math.floor(Math.random() * possibleResponses.length)];
  }
  for (const phrase in cannedResponses) {
    if (lowerUserInput.includes(phrase)) {
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
  const isShortAnswer = userInput.split(/\s+/).length <= 2;
  const sentenceLength = isShortAnswer ? (Math.floor(Math.random() * 2) + 2) : (Math.floor(Math.random() * 6) + 4); // 2-3 words for short, 4-9 for long

  for (let i = 0; i < sentenceLength; i++) {
    const possibleNextWords = markovChains[currentWord] || vocabulary;
    const nextWord = possibleNextWords[Math.floor(Math.random() * possibleNextWords.length)];
    
    // Avoid immediate repetition
    if (response[response.length - 1] !== nextWord) {
        response.push(nextWord);
    }
    
    currentWord = nextWord;
    if (!markovChains[currentWord]) {
        // If we hit a word with no chain, try to find another one from vocabulary
        currentWord = vocabulary[Math.floor(Math.random() * vocabulary.length)];
        if (i < sentenceLength - 1) { // dont break on the last word
           break;
        }
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
