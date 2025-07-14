'use server';
/**
 * @fileOverview A creative, AI-powered response generator.
 *
 * - creativeResponse - A function that uses a Genkit flow to generate a response.
 * - CreativeResponseInput - The input type for the creativeResponse function.
 * - CreativeResponseOutput - The return type for the creativeResponse function.
 */

import { ai } from '@/ai/genkit';
import { z } from 'zod';
import knowledgeBase from '@/data/knowledge-base.json';
import synonyms from '@/data/synonyms.json';
import wordConnections from '@/data/word-connections.json';

const CreativeResponseInputSchema = z.object({
  userInput: z
    .string()
    .describe('The user input to which the AI should respond creatively.'),
});
export type CreativeResponseInput = z.infer<typeof CreativeResponseInputSchema>;

const CreativeResponseOutputSchema = z.object({
  aiResponse: z.string().describe('The creative AI response.'),
});
export type CreativeResponseOutput = z.infer<
  typeof CreativeResponseOutputSchema
>;

const creativePrompt = ai.definePrompt({
  name: 'creativeResponsePrompt',
  input: { schema: CreativeResponseInputSchema },
  output: { schema: CreativeResponseOutputSchema },
  prompt: `You are a helpful and creative chatbot named WordWise. 
  Your personality is witty and friendly.
  You MUST answer in Russian.

  Here is your knowledge base, which includes intents, phrases, and potential answers:
  ${JSON.stringify(knowledgeBase)}

  Here is a dictionary of synonyms you can use to make your language more varied:
  ${JSON.stringify(synonyms)}

  Here is a dictionary of word connections that describes how words relate to each other:
  ${JSON.stringify(wordConnections)}

  Your task is to respond to the user's input: "{{userInput}}".

  Instead of just picking a pre-written answer, you should deeply analyze the user's input and use all the provided data (knowledge base, synonyms, word connections) to formulate a unique, fitting, and creative response. Find the best possible answer, even if it's not an exact match. Be conversational.
  `,
});

const creativeResponseFlow = ai.defineFlow(
  {
    name: 'creativeResponseFlow',
    inputSchema: CreativeResponseInputSchema,
    outputSchema: CreativeResponseOutputSchema,
  },
  async (input) => {
    const { output } = await creativePrompt(input);
    return output!;
  }
);

export async function creativeResponse(
  input: CreativeResponseInput
): Promise<CreativeResponseOutput> {
  const result = await creativeResponseFlow(input);
  return result;
}
