'use server';
/**
 * @fileOverview A contextual response AI bot based on a knowledge base, synonyms, and word connections.
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
  model: z.enum(['R', 'Q']).describe('The model to use: R for Rigid, Q for Creative.'),
  history: z.array(z.string()).optional().describe('The last 3 messages in the conversation.'),
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

// --- Type Definitions ---
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

const kb = knowledgeBase as KnowledgeBase;
const syn = synonyms as Synonyms;
const wc = wordConnections as WordConnections;

const defaultResponses = kb['неизвестная_фраза'].ответы;

// --- Bot's "Brain" ---

/**
 * A simple and safe calculator function to evaluate arithmetic expressions.
 * It respects parentheses and order of operations (*, / before +, -).
 * @param expression The mathematical expression to evaluate.
 * @returns The result of the calculation or null if the expression is invalid.
 */
function calculateExpression(expression: string): number | null {
  try {
    // This is a safer alternative to eval(). It creates a new Function
    // which is executed in a closed scope. It's safe as long as the
    // expression only contains numbers and basic math operators.
    // We validate the expression with a regex to ensure safety.
    const sanitizedExpression = expression.replace(/[^-()\d/*+.]/g, '');
    if (sanitizedExpression !== expression) {
      // Invalid characters were found
      return null;
    }

    // eslint-disable-next-line no-new-func
    const result = new Function(`return ${expression}`)();

    if (typeof result !== 'number' || !isFinite(result)) {
      return null;
    }
    return result;
  } catch (error) {
    return null;
  }
}


/**
 * Replaces words in a sentence with their synonyms to make it more dynamic.
 * @param sentence The sentence to process.
 * @returns A sentence with some words replaced by synonyms.
 */
function synonymize(sentence: string): string {
  const words = sentence.split(/(\s+|,|\.|\?|!)/);
  const newWords = words.map(word => {
    const lowerWord = word.toLowerCase();
    const synonymList = syn[lowerWord];
    if (synonymList && Math.random() > 0.5) { // 50% chance to replace
      const randomSynonym =
        synonymList[Math.floor(Math.random() * synonymList.length)];
      // Preserve capitalization
      if (word.length > 0 && word[0] === word[0].toUpperCase()) {
        return randomSynonym.charAt(0).toUpperCase() + randomSynonym.slice(1);
      }
      return randomSynonym;
    }
    return word;
  });
  return newWords.join('');
}


/**
 * Model R (Rigid): Generates a response based on strict keyword matching from the knowledge base.
 * @param userInput The user's message.
 * @param history The conversation history.
 * @returns A response string.
 */
function generateRigidResponse(userInput: string, history: string[] = []): string {
    // 1. Check for a mathematical expression first.
    const mathRegex = /^[0-9\s\+\-\*\/\(\)\.]+$/;
    if (mathRegex.test(userInput.trim())) {
      const result = calculateExpression(userInput);
      if (result !== null) {
        return `Результат: ${result}`;
      }
    }

    const lowerCaseInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
    const wordsInInput = new Set(lowerCaseInput.split(/\s+/).filter(w => w));

    let bestMatch: { intent: string; score: number } | null = null;

    // Calculate best match from knowledge base using Jaccard similarity on the LATEST user input first
    for (const intent in kb) {
        if (intent === 'неизвестная_фраза' || !Object.prototype.hasOwnProperty.call(kb, intent)) continue;

        const intentData = kb[intent];
        for (const phrase of intentData.фразы) {
            const lowerPhrase = phrase.toLowerCase();
            const phraseWords = new Set(lowerPhrase.split(/\s+/));
            
            const intersection = new Set([...phraseWords].filter(x => wordsInInput.has(x)));
            const union = new Set([...phraseWords, ...wordsInInput]);
            const score = union.size > 0 ? intersection.size / union.size : 0;

            if (score > (bestMatch?.score ?? 0)) {
                bestMatch = { intent, score };
            }
        }
    }

    // Decide if the match is good enough
    if (bestMatch && bestMatch.score > 0.1) { // Threshold can be adjusted
        const responses = kb[bestMatch.intent].ответы;
        const randomIndex = Math.floor(Math.random() * responses.length);
        return synonymize(responses[randomIndex]);
    }

    // Fallback to default response if no good match is found
    const randomIndex = Math.floor(Math.random() * defaultResponses.length);
    return synonymize(defaultResponses[randomIndex]);
}


/**
 * Model Q (Creative): Generates a response by finding the best matching intent or using word connections.
 * It uses a scoring mechanism to find the "ideal" response.
 * @param userInput The user's message.
 * @param history The conversation history.
 * @returns A response string.
 */
function generateCreativeResponse(userInput: string, history: string[] = []): string {
    const lowerCaseInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
    const wordsInInput = new Set(lowerCaseInput.split(/\s+/).filter(w => w));

    let bestMatch: { intent: string; score: number } | null = null;

    // 1. Calculate best match from knowledge base
    for (const intent in kb) {
        if (intent === 'неизвестная_фраза' || !Object.prototype.hasOwnProperty.call(kb, intent)) continue;

        const intentData = kb[intent];
        let highestScoreForIntent = 0;

        for (const phrase of intentData.фразы) {
            const lowerPhrase = phrase.toLowerCase();
            const phraseWords = new Set(lowerPhrase.split(/\s+/));
            const intersection = new Set([...phraseWords].filter(x => {
                // Check for partial matches (e.g., "привет" in "приветик")
                for (const inputWord of wordsInInput) {
                    if (inputWord.includes(x) || x.includes(inputWord)) {
                        return true;
                    }
                }
                return false;
            }));

            const union = new Set([...phraseWords, ...wordsInInput]);
            const score = union.size > 0 ? intersection.size / union.size : 0; // Jaccard similarity
            
            if (score > highestScoreForIntent) {
                highestScoreForIntent = score;
            }
        }

        if (highestScoreForIntent > 0 && (!bestMatch || highestScoreForIntent > bestMatch.score)) {
            bestMatch = { intent, score: highestScoreForIntent };
        }
    }

    // 2. Decide if the match is good enough
    if (bestMatch && bestMatch.score > 0.1) {
        const responses = kb[bestMatch.intent].ответы;
        const randomIndex = Math.floor(Math.random() * responses.length);
        return synonymize(responses[randomIndex]);
    }

    // 3. If no good match, ALWAYS try to generate from word connections.
    const connectionResponse = generateConnectionResponse(userInput);
    
    // 4. Fallback to a creative default if no connection is found
    if (connectionResponse) {
        return synonymize(connectionResponse);
    }
    
    return synonymize("Хм, интересный вопрос. Дай-ка подумать...");
}

/**
 * Helper for Model Q. Generates a response based on word connections if a direct intent is not found.
 * @param userInput The user's message.
 * @returns A generated response string or null if no connection is found.
 */
function generateConnectionResponse(userInput: string): string | null {
  const lowerCaseInput = userInput.toLowerCase();
  const words = lowerCaseInput.split(/\s+/).filter(w => w.length > 0);
  const connections = wc.словарь_связей;

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

  for (const word of words) {
    for (const partOfSpeech in connections) {
       const wordsInPOS = connections[partOfSpeech as keyof typeof connections];
       for (const mainWord in wordsInPOS) {
        const properties = wordsInPOS[mainWord as keyof typeof wordsInPOS];
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


// --- Main Entry Point ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  let aiResponse: string;
  const history = input.history || [];

  if (input.model === 'Q') {
    aiResponse = generateCreativeResponse(input.userInput, history);
  } else {
    aiResponse = generateRigidResponse(input.userInput, history);
  }

  return {aiResponse};
}
