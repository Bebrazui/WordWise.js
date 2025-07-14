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
import wordConnections from '@/data/word-connections.json';

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

type WordConnections = {
  словарь_связей: {
    [partOfSpeech: string]: {
      [word: string]: {
        [relation: string]: string | string[];
      };
    };
  };
};

const defaultResponses = (knowledgeBase as KnowledgeBase)['неизвестная_фраза']
  .ответы;

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
    if (synonymList && Math.random() > 0.5) {
      // 50% chance to replace
      const randomSynonym =
        synonymList[Math.floor(Math.random() * synonymList.length)];
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
 * Generates a response based on word connections if a direct intent is not found.
 * @param userInput The user's message.
 * @returns A generated response string or null if no connection is found.
 */
function generateConnectionResponse(userInput: string): string | null {
  const lowerCaseInput = userInput.toLowerCase();
  const words = lowerCaseInput.split(/\s+/).filter(w => w.length > 2); // Get individual words
  const connections = (wordConnections as WordConnections).словарь_связей;

  for (const word of words) {
    for (const partOfSpeech in connections) {
      const wordsInPOS = connections[partOfSpeech as keyof typeof connections];
      if (Object.prototype.hasOwnProperty.call(wordsInPOS, word)) {
        const properties = wordsInPOS[word as keyof typeof wordsInPOS];
        const isA = properties.is_a;
        const canDo = properties.can_do;

        let response = `${word.charAt(0).toUpperCase() + word.slice(1)}`;

        if (isA) {
          response += ` - это ${
            Array.isArray(isA) ? isA.join(', ') : isA
          }.`;
        } else {
          response += '.';
        }

        if (canDo) {
          response += ` Может ${
            Array.isArray(canDo) ? canDo.join(', ') : canDo
          }.`;
        }

        return response;
      }
    }
  }

  // Check if any word is a value within the connections
   for (const word of words) {
    for (const partOfSpeech in connections) {
       const wordsInPOS = connections[partOfSpeech as keyof typeof connections];
       for (const mainWord in wordsInPOS) {
        const properties = wordsInPOS[mainWord];
        for (const prop in properties) {
            const values = properties[prop as keyof typeof properties];
            if(Array.isArray(values) && values.includes(word)) {
                return `${word.charAt(0).toUpperCase() + word.slice(1)} - это один из примеров для '${mainWord}'.`;
            }
        }
       }
    }
  }


  return null;
}

/**
 * This function uses keyword matching to generate a response from a structured knowledge base.
 * It finds a matching intent and returns a random, synonymized response from that intent's list of answers.
 * If no intent is found, it tries to generate a response from word connections.
 * @param userInput The user's message.
 * @returns A response string.
 */
function generateResponse(userInput: string): string {
  const lowerCaseInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
  const wordsInInput = lowerCaseInput.split(/\s+/);

  const intents = Object.keys(knowledgeBase) as Array<
    keyof typeof knowledgeBase
  >;

  let bestMatch: { intent: string, score: number } | null = null;

  for (const intent of intents) {
    if (intent === 'неизвестная_фраза') continue;

    const intentData = (knowledgeBase as KnowledgeBase)[intent];
    
    // Check for exact or very close match first
    const perfectMatch = intentData.фразы.find(phrase => lowerCaseInput === phrase.toLowerCase());
    if(perfectMatch){
        bestMatch = { intent, score: 100 };
        break; // Perfect match found, no need to check others
    }

    // If not a perfect match, check for keywords
    let score = 0;
    intentData.фразы.forEach(phrase => {
        const phraseWords = phrase.toLowerCase().split(/\s+/);
        const commonWords = phraseWords.filter(word => wordsInInput.includes(word));
        // Simple scoring: prioritize longer matching phrases
        if (commonWords.length > 0) {
            score = Math.max(score, commonWords.length / phraseWords.length);
        }
    });

    if (bestMatch === null || score > bestMatch.score) {
        if(score > 0.5) { // Require a reasonable match
            bestMatch = { intent, score };
        }
    }
  }

  if (bestMatch) {
    const intentData = (knowledgeBase as KnowledgeBase)[bestMatch.intent];
    const responses = intentData.ответы;
    const randomIndex = Math.floor(Math.random() * responses.length);
    const chosenResponse = responses[randomIndex];
    return synonymize(chosenResponse);
  }


  // If no intent was matched, try to generate from connections
  const connectionResponse = generateConnectionResponse(userInput);
  if (connectionResponse) {
    return synonymize(connectionResponse);
  }

  // If still no response, return a random default response
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
