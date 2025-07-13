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

// A very simple knowledge base.
// The key is the keyword to look for, and the value is the response.
const knowledgeBase: {[key: string]: string} = {
    'привет': 'Привет! Чем я могу помочь?',
    'здравствуй': 'Здравствуй!',
    'пока': 'До свидания!',
    'до свидания': 'Пока! Удачи!',
    'как дела': 'У меня все хорошо, я же программа. А у тебя?',
    'кто ты': 'Я — WordWise, очень простой бот.',
    'спасибо': 'Пожалуйста!',
    'помощь': 'Я еще не умею помогать, но я учусь.',
    'марс': 'Марс - это красная планета.',
    'погода': 'Я не знаю какая погода, я же в компьютере.',
    'шутку': 'Колобок повесился.',
};

// A default response if no keyword is found.
const defaultResponse = 'Интересная мысль. Я не совсем понимаю.';

/**
 * This is a very primitive function to generate a response.
 * It finds the first keyword from the knowledge base that exists in the user input.
 * @param userInput The user's message.
 * @returns A response string.
 */
function generateResponse(userInput: string): string {
  const lowerCaseInput = userInput.toLowerCase();

  for (const keyword in knowledgeBase) {
    if (lowerCaseInput.includes(keyword)) {
      return knowledgeBase[keyword];
    }
  }

  return defaultResponse;
}

// --- End of the bot's "brain" ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  
  const aiResponse = generateResponse(input.userInput);

  return { aiResponse };
}
