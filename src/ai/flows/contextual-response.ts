'use server';
/**
 * @fileOverview A contextual response AI bot.
 *
 * - contextualResponse - A function that handles the response generation process.
 * - ContextualResponseInput - The input type for the contextualResponse function.
 * - ContextualResponseOutput - The return type for the contextualResponse function.
 */

import {z} from 'zod';
import knowledgeBase from '@/data/knowledge-base.json';


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

// A default response if no keyword is found.
const defaultResponse = 'Интересная мысль. Я не совсем понимаю.';

/**
 * This function uses simple keyword matching to generate a response.
 * @param userInput The user's message.
 * @returns A response string.
 */
function generateResponse(userInput: string): string {
  const lowerCaseInput = userInput.toLowerCase();
  
  // Find a keyword that is included in the user's input.
  for (const keyword in knowledgeBase) {
    if (lowerCaseInput.includes(keyword)) {
      return (knowledgeBase as Record<string, string>)[keyword];
    }
  }

  return defaultResponse;
}

// --- End of the bot's "brain" ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  const aiResponse = generateResponse(input.userInput);

  return {aiResponse};
}
