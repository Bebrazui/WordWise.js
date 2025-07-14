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

// --- State for avoiding repetition ---
const lastResponseMap = new Map<string, string>();


// --- Bot's "Brain" ---

/**
 * A simple and safe calculator function to evaluate arithmetic expressions.
 * It respects parentheses and order of operations (*, / before +, -).
 * @param expression The mathematical expression to evaluate.
 * @returns The result of the calculation or null if the expression is invalid.
 */
function calculateExpression(expression: string): number | null {
  try {
    // Improved regex to better extract expressions from text
    const sanitizedExpression = expression.replace(/[^-()\d/*+.]/g, ' ').trim();
    if (!sanitizedExpression) {
      return null;
    }
    
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
 * Increased chance to synonymize.
 * @param sentence The sentence to process.
 * @returns A sentence with some words replaced by synonyms.
 */
function synonymize(sentence: string): string {
  const words = sentence.split(/(\s+|,|\.|\?|!)/);
  const newWords = words.map(word => {
    const lowerWord = word.toLowerCase();
    const synonymList = syn[lowerWord];
    if (synonymList && Math.random() < 0.75) { // 75% chance to replace
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
    const mathExpressionRegex = /(?:реши|посчитай|сколько будет)\s*([0-9+\-*/().\s]+)|((?:\d|\.)+\s*[+\-*/]\s*(?:\d|\.)+)/gi;
    const match = mathExpressionRegex.exec(userInput);

    if (match) {
        const expression = match[1] || match[2];
        if (expression) {
            const result = calculateExpression(expression);
            if (result !== null) {
                return `Результат: ${result}`;
            }
        }
    }

    const lowerCaseInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
    const wordsInInput = new Set(lowerCaseInput.split(/\s+/).filter(w => w));

    let bestMatch: { intent: string; score: number } | null = null;
    
    // Use history for context, but prioritize the current user input.
    const contextText = [...history, userInput].join(' ');
    const lowerCaseContext = contextText.toLowerCase().replace(/[.,!?]/g, '');
    const wordsInContext = new Set(lowerCaseContext.split(/\s+/).filter(w => w));

    // Calculate best match from knowledge base
    for (const intent in kb) {
        if (intent === 'неизвестная_фраза' || !Object.prototype.hasOwnProperty.call(kb, intent)) continue;

        const intentData = kb[intent];
        let maxScoreForIntent = 0;

        // Score based on direct match with user's last message
        for (const phrase of intentData.фразы) {
            const lowerPhrase = phrase.toLowerCase();
            const phraseWords = new Set(lowerPhrase.split(/\s+/).filter(Boolean));
            
            const intersection = new Set([...phraseWords].filter(x => wordsInInput.has(x)));
            const union = new Set([...phraseWords, ...wordsInInput]);
            const score = union.size > 0 ? intersection.size / union.size : 0;
            
            if (score > maxScoreForIntent) {
                maxScoreForIntent = score;
            }
        }
        
        // Boost score based on context from history
        const contextIntersection = new Set([...intentData.фразы.flatMap(p => p.toLowerCase().split(/\s+/))].filter(x => wordsInContext.has(x)));
        const contextScore = wordsInContext.size > 0 ? contextIntersection.size / wordsInContext.size : 0;

        // Give much higher weight to the last message
        const finalScore = (maxScoreForIntent * 0.8) + (contextScore * 0.2);

        if (finalScore > (bestMatch?.score ?? 0)) {
            bestMatch = { intent, score: finalScore };
        }
    }

    // Adjust threshold for better accuracy
    if (bestMatch && bestMatch.score > 0.15) {
        let responses = kb[bestMatch.intent].ответы;
        // Avoid repeating the last response for this intent
        const lastResponse = lastResponseMap.get(bestMatch.intent);
        if (lastResponse && responses.length > 1) {
            responses = responses.filter(r => r !== lastResponse);
        }
        const response = responses[Math.floor(Math.random() * responses.length)];
        lastResponseMap.set(bestMatch.intent, response);
        return synonymize(response);
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

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  return matrix[b.length][a.length];
}

const personalPronounMarkers = ["я", "у меня", "мой", "мне", "я сам"];
const questionAboutWellbeing = ["как дела", "как ты", "как поживаешь", "как твое", "как сам", "как настроение"];
const positiveUserStates = ["хорошо", "нормально", "отлично", "замечательно", "прекрасно", "в порядке", "ничего", "пойдет"];

/**
 * Checks if the user is likely responding to a question about their well-being.
 * @param userInput The user's current message.
 * @param history The conversation history.
 * @returns true if it's likely a personal response to a well-being question.
 */
function isPersonalResponseToWellbeing(userInput: string, history: string[]): boolean {
    const lowerInput = userInput.toLowerCase();
    const lastBotMessage = history.length > 0 ? history[history.length - 1].toLowerCase() : "";

    const wasAsked = questionAboutWellbeing.some(q => lastBotMessage.includes(q));
    if (!wasAsked) {
        return false;
    }

    const isPersonal = personalPronounMarkers.some(marker => lowerInput.startsWith(marker));
    const isPositiveState = positiveUserStates.includes(lowerInput.replace(/[.,!?]/g, '').trim());

    return isPersonal || isPositiveState;
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

  // 1. Check for specific conversational contexts, like responding to "how are you?"
  if (isPersonalResponseToWellbeing(userInput, history)) {
    // This is a direct response to a question about the user's wellbeing.
    const suitableResponses = [
        "Рад это слышать!",
        "Отлично! Чем теперь займемся?",
        "Здорово! Если что-то понадобится, я здесь.",
        "Это хорошо. Что дальше по плану?",
     ];
     return synonymize(suitableResponses[Math.floor(Math.random() * suitableResponses.length)]);
  }

  // 2. Try to find a direct or fuzzy match in the knowledge base first.
  // This version will check for partial matches within the user input.
  let bestMatch: { intent: string; score: number } | null = null;
  for (const intent in kb) {
    if (intent === 'неизвестная_фраза' || !Object.prototype.hasOwnProperty.call(kb, intent)) continue;

    const intentData = kb[intent];
    for (const phrase of intentData.фразы) {
      const lowerPhrase = phrase.toLowerCase();
      
      // Check if the known phrase is a substring of the user's input
      if (lowerCaseInput.includes(lowerPhrase)) {
        const score = lowerPhrase.length / lowerCaseInput.length; // Simple score based on length ratio
        if (score > (bestMatch?.score ?? 0)) {
          bestMatch = { intent, score };
        }
      } else {
        // Fallback to Levenshtein for typos if no substring match
        const distance = levenshteinDistance(lowerCaseInput, lowerPhrase);
        const threshold = Math.floor(lowerPhrase.length / 3);
        if (distance <= threshold) {
            const score = (1 - (distance / lowerPhrase.length)) * 0.9; // Lower score for fuzzy matches
            if (score > (bestMatch?.score ?? 0)) {
              bestMatch = { intent, score };
            }
        }
      }
    }
  }

  // If a reasonably good match is found, use it.
  if (bestMatch && bestMatch.score > 0.5) {
    let responses = kb[bestMatch.intent].ответы;
    const lastResponse = lastResponseMap.get(bestMatch.intent);
    if (lastResponse && responses.length > 1) {
        responses = responses.filter(r => r !== lastResponse);
    }
    const response = responses[Math.floor(Math.random() * responses.length)];
    lastResponseMap.set(bestMatch.intent, response);
    // If the match was partial, we might slightly alter the response
    if (bestMatch.score < 1.0) {
      return synonymize(response) + " А что ты еще хотел узнать?";
    }
    return synonymize(response);
  }

  // 3. If no direct match, get creative with word connections.
  const connectionResponse = generateConnectionResponse(userInput);
  if (connectionResponse) {
    return synonymize(connectionResponse);
  }

  // 4. Fallback to a more creative, thoughtful default response.
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

  const allWordsAndSynonyms = new Set(words);
  words.forEach(word => {
    if (syn[word]) {
      syn[word].forEach(s => allWordsAndSynonyms.add(s));
    }
  });

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
    return null;
  }

  if (foundFacts.length === 1) {
    return foundFacts[0].fact;
  }
  
  let combinedResponse = "Если я правильно тебя понимаю, ты говоришь о нескольких вещах. ";
  const factStrings = foundFacts.map(f => f.fact.replace(/.$/, ''));
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
