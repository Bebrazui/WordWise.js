'use server';

/**
 * @fileOverview This file defines a flow for generating contextually relevant responses.
 *
 * - contextualResponse - A function that takes user input and returns a contextually relevant response.
 * - ContextualResponseInput - The input type for the contextualResponse function.
 * - ContextualResponseOutput - The return type for the contextualResponse function.
 */

import {ai} from '@/ai/genkit';
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

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  return contextualResponseFlow(input);
}

const prompt = ai.definePrompt({
  name: 'contextualResponsePrompt',
  input: {schema: ContextualResponseInputSchema},
  output: {schema: ContextualResponseOutputSchema},
  prompt: `You are a simple chatbot named WordWise. Your vocabulary is limited, similar to a person learning a new language. You are friendly and try to be helpful, but your answers are sometimes a bit strange or grammatically incorrect because of your limited knowledge. You should never say that you are an AI. You should always answer in Russian.

User's message: {{{userInput}}}`,
});

const contextualResponseFlow = ai.defineFlow(
  {
    name: 'contextualResponseFlow',
    inputSchema: ContextualResponseInputSchema,
    outputSchema: ContextualResponseOutputSchema,
  },
  async (input) => {
    const {output} = await prompt(input);
    return output!;
  }
);
