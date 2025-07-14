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
import synonyms from '@/data/synonyms.json';

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

type KnowledgeBase = {
  [intent: string]: {
    фразы: string[];
    ответы: string[];
  };
};

type Synonyms = {
  [key: string]: string[];
};

const defaultResponses = (knowledgeBase as KnowledgeBase)['неизвестная_фраза'].ответы;

/**
 * Replaces words in a sentence with their synonyms to make it more dynamic.
 * @param sentence The sentence to process.
 * @returns A sentence with some words replaced by synonyms.
 */
function synonymize(sentence: string): string {
  const words = sentence.split(/(\s+|,|\.|\?|!)/);
  const newWords = words.map(word => {
    const lowerWord = word.toLowerCase();
    const synonymList = (synonyms as Synonyms)[lowerWord];
    if (synonymList && Math.random() > 0.5) { // 50% chance to replace
      const randomSynonym = synonymList[Math.floor(Math.random() * synonymList.length)];
      // Preserve capitalization
      if (word[0] === word[0].toUpperCase()) {
        return randomSynonym.charAt(0).toUpperCase() + randomSynonym.slice(1);
      }
      return randomSynonym;
    }
    return word;
  });
  return newWords.join('');
}

/**
 * This function uses keyword matching to generate a response from a structured knowledge base.
 * It finds a matching intent and returns a random, synonymized response from that intent's list of answers.
 * @param userInput The user's message.
 * @returns A response string.
 */
function generateResponse(userInput: string): string {
  const lowerCaseInput = userInput.toLowerCase();
  
  const intents = Object.keys(knowledgeBase) as Array<keyof typeof knowledgeBase>;

  for (const intent of intents) {
    // Skip the default case, we'll handle it last
    if (intent === 'неизвестная_фраза') continue;

    const intentData = (knowledgeBase as KnowledgeBase)[intent];
    const foundPhrase = intentData.фразы.find(phrase => lowerCaseInput.includes(phrase.toLowerCase()));

    if (foundPhrase) {
      const responses = intentData.ответы;
      // Return a random response from the list
      const randomIndex = Math.floor(Math.random() * responses.length);
      const chosenResponse = responses[randomIndex];
      return synonymize(chosenResponse);
    }
  }

  // If no intent was matched, return a random default response
  const randomIndex = Math.floor(Math.random() * defaultResponses.length);
  const chosenResponse = defaultResponses[randomIndex];
  return synonymize(chosenResponse);
}

// --- End of the bot's "brain" ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  const aiResponse = generateResponse(input.userInput);

  return {aiResponse};
}

    