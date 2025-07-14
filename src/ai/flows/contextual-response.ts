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
    const sanitizedExpression = expression.replace(/[^-()\d/*+.]/g, ' ').trim();
    if (!sanitizedExpression) {
      return null;
    }
    
    // Check for invalid characters that weren't just whitespace
    if (/[^-()\d/*+.\s]/.test(expression)) {
        // if there are other characters than math and spaces, it's not a pure expression
    }

    // To be safe, ensure the sanitized expression is valid before executing
    // This regex checks for a valid structure, preventing things like `2+` or `*3`
    const validMathRegex = /^(?:-?\d+(?:\.\d+)?(?:[+\-*/]\s?-?\s?\d+(?:\.\d+)?)*?)$/;
    // A more complex one to handle parentheses is tricky. For now, let's keep it simple
    // and let the Function constructor handle errors.

    // eslint-disable-next-line no-new-func
    const result = new Function(`return ${sanitizedExpression}`)();

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
    // This regex extracts potential math expressions from the text.
    const mathExpressionRegex = /((?:\d|\.)+\s*[+\-*/]\s*(?:\d|\.)+)/g;
    const potentialExpression = userInput.match(mathExpressionRegex)?.[0];

    if (potentialExpression) {
        const result = calculateExpression(potentialExpression);
        if (result !== null) {
            return `Результат: ${result}`;
        }
    }

    const lowerCaseInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
    const wordsInInput = new Set(lowerCaseInput.split(/\s+/).filter(w => w));

    let bestMatch: { intent: string; score: number } | null = null;
    
    // Combine history with current input for context, but prioritize current input for matching.
    const textToAnalyze = [userInput, ...history].join(' ');
    const lowerCaseText = textToAnalyze.toLowerCase().replace(/[.,!?]/g, '');
    const wordsInText = new Set(lowerCaseText.split(/\s+/).filter(w => w));


    // Calculate best match from knowledge base
    for (const intent in kb) {
        if (intent === 'неизвестная_фраза' || !Object.prototype.hasOwnProperty.call(kb, intent)) continue;

        const intentData = kb[intent];
        let highestScoreForIntent = 0;

        for (const phrase of intentData.фразы) {
            const lowerPhrase = phrase.toLowerCase();
            const phraseWords = new Set(lowerPhrase.split(/\s+/).filter(Boolean));
            
            // Check for partial matches (e.g., "привет" in "приветик")
            const intersection = new Set([...phraseWords].filter(x => {
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
        
        // Boost score if the intent is more relevant to the whole conversation context
        const contextIntersection = new Set([...wordsInText].filter(x => {
             for (const phrase of intentData.фразы) {
                if(phrase.toLowerCase().includes(x)) return true;
             }
             return false;
        }));
        const contextScore = contextIntersection.size / wordsInText.size;


        const finalScore = (highestScoreForIntent * 0.7) + (contextScore * 0.3);

        if (finalScore > (bestMatch?.score ?? 0)) {
            bestMatch = { intent, score: finalScore };
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
 * Calculates the Levenshtein distance between two strings.
 * This is a measure of the difference between two sequences.
 * @param a The first string.
 * @param b The second string.
 * @returns The Levenshtein distance.
 */
function levenshteinDistance(a: string, b: string): number {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  const matrix = [];

  // increment along the first column of each row
  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }

  // increment each column in the first row
  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  // Fill in the rest of the matrix
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1, // substitution
          matrix[i][j - 1] + 1,     // insertion
          matrix[i - 1][j] + 1      // deletion
        );
      }
    }
  }

  return matrix[b.length][a.length];
}


/**
 * Model Q (Creative): Generates a response by finding the best matching intent or using word connections.
 * It uses a scoring mechanism to find the "ideal" response and handles typos.
 * @param userInput The user's message.
 * @param history The conversation history.
 * @returns A response string.
 */
function generateCreativeResponse(userInput: string, history: string[] = []): string {
  const lowerCaseInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
  const wordsInInput = lowerCaseInput.split(/\s+/).filter(w => w);

  // 1. Try to find a direct or fuzzy match in the knowledge base first
  let bestMatch: { intent: string; score: number } | null = null;
  for (const intent in kb) {
    if (intent === 'неизвестная_фраза' || !Object.prototype.hasOwnProperty.call(kb, intent)) continue;

    const intentData = kb[intent];
    for (const phrase of intentData.фразы) {
      const lowerPhrase = phrase.toLowerCase();
      const phraseWords = lowerPhrase.split(/\s+/).filter(Boolean);
      let currentScore = 0;
      
      for(const inputWord of wordsInInput) {
        for(const phraseWord of phraseWords) {
            // Allow for a small number of typos (Levenshtein distance)
            // The threshold can be adjusted. e.g. 1 for short words, 2 for longer.
            const distance = levenshteinDistance(inputWord, phraseWord);
            const threshold = phraseWord.length > 4 ? 2 : 1;
            if (distance <= threshold) {
                currentScore++;
            }
        }
      }

      if (currentScore > (bestMatch?.score ?? 0)) {
        bestMatch = { intent, score: currentScore };
      }
    }
  }

  // If a reasonably good match is found, use it. This makes the bot feel responsive to direct questions.
  if (bestMatch && bestMatch.score > 0) {
    const responses = kb[bestMatch.intent].ответы;
    const randomIndex = Math.floor(Math.random() * responses.length);
    return synonymize(responses[randomIndex]);
  }

  // 2. If no direct match, get creative with word connections.
  const connectionResponse = generateConnectionResponse(userInput);
  if (connectionResponse) {
    return synonymize(connectionResponse);
  }

  // 3. Fallback to a more creative, thoughtful default response.
  // This avoids the "I don't know" trap.
  const thoughtfulResponses = [
    "Это интересный поворот. Дай мне секунду, чтобы обдумать это.",
    "Хм, ты затронул любопытную тему. Я пытаюсь найти связи...",
    "Твой вопрос заставляет меня взглянуть на вещи под другим углом.",
    `Я размышляю над словом "${wordsInInput[0] || 'это'}"... и что оно может означать в этом контексте.`,
    "Это сложный вопрос. Мои алгоритмы ищут наиболее подходящий ответ."
  ];
  const randomIndex = Math.floor(Math.random() * thoughtfulResponses.length);
  return synonymize(thoughtfulResponses[randomIndex]);
}

/**
 * Helper for Model Q. Generates a response based on word connections if a direct intent is not found.
 * This version is more "creative" and tries to combine information.
 * @param userInput The user's message.
 * @returns A generated response string or null if no connection is found.
 */
function generateConnectionResponse(userInput: string): string | null {
  const lowerCaseInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
  const words = [...new Set(lowerCaseInput.split(/\s+/).filter(w => w.length > 2))]; // Use unique words, ignore short ones
  const connections = wc.словарь_связей;
  
  const foundFacts: { word: string, fact: string }[] = [];

  // Get all synonyms to broaden the search
  const allWordsAndSynonyms = new Set(words);
  words.forEach(word => {
    if (syn[word]) {
      syn[word].forEach(s => allWordsAndSynonyms.add(s));
    }
  });

  // Find all possible connections for the words in the input
  for (const word of allWordsAndSynonyms) {
    for (const partOfSpeech in connections) {
      const wordsInPOS = connections[partOfSpeech as keyof typeof connections];
      if (Object.prototype.hasOwnProperty.call(wordsInPOS, word)) {
        const properties = wordsInPOS[word as keyof typeof wordsInPOS];
        let fact = `${word.charAt(0).toUpperCase() + word.slice(1)}`;

        if (properties.is_a) {
          fact += ` - это ${Array.isArray(properties.is_a) ? properties.is_a[0] : properties.is_a}`;
        }
        if (properties.can_do) {
           fact += `, который может ${Array.isArray(properties.can_do) ? properties.can_do[0] : properties.can_do}.`;
        } else {
           fact += ".";
        }
        foundFacts.push({ word, fact });
      }
    }
  }

  if (foundFacts.length === 0) {
    return null; // No connections found
  }

  if (foundFacts.length === 1) {
    return foundFacts[0].fact; // Only one fact, return it
  }
  
  // If multiple facts are found, try to combine them into a coherent sentence
  let combinedResponse = "Если я правильно тебя понимаю, ты говоришь о нескольких вещах. ";
  const factStrings = foundFacts.map(f => f.fact.replace(/.$/, '')); // remove trailing period
  combinedResponse += factStrings.join(' и ');
  combinedResponse += ". Это довольно интересное сочетание.";

  return combinedResponse;
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
