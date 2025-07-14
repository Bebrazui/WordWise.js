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
import learnedWords from '@/data/learned-words.json';

// NOTE: This is a simplified in-memory "database" for the sake of this example.
// In a real-world application, you would use a proper database like Firestore
// to persist learned data across server restarts and multiple instances.
const fs = {
  writeFile: (path: string, data: string, cb: (err?: any) => void) => {
    // This is a mock. In a real environment, you'd need Node's 'fs' module
    // and write permissions, which are not available in this context.
    console.log(`MOCK WRITE to ${path}: ${data.substring(0, 100)}...`);
    cb();
  },
};
import path from 'path';

const ContextualResponseInputSchema = z.object({
  userInput: z
    .string()
    .describe('The user input to which the AI should respond.'),
  history: z
    .array(z.string())
    .optional()
    .describe('The last 3 messages in the conversation.'),
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
  веса_частей_речи: {
    [partOfSpeech: string]: number;
  };
  словарь_связей: {
    [partOfSpeech: string]: {
      [word: string]: {
        [relation: string]: string | string[];
      };
    };
  };
};

type LearnedWords = {
  [word: string]: string;
};

const kb = knowledgeBase as KnowledgeBase;
const syn = synonyms as Synonyms;
const wc = wordConnections as WordConnections;
const lw = learnedWords as LearnedWords;

// --- State for avoiding repetition and simulating learning ---
const lastResponseMap = new Map<string, string>();
const sessionMemory = {
  lastUnknownWord: null as string | null,
};

// --- Bot's "Brain" (Model Q - Quantum) ---

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
    if (synonymList && Math.random() < 0.75) { // Increased chance to 75%
      const randomSynonym =
        synonymList[Math.floor(Math.random() * synonymList.length)];
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


// --- Pipeline Stage 1: Direct Response Handler ---
/**
 * Handles simple, direct responses to bot's questions, like "нет" to "Я правильно тебя понял?".
 * This is the first check to avoid misinterpretation loops.
 * @param userInput The user's message.
 * @param history The conversation history.
 * @returns A specific response or null if no direct response context is found.
 */
function handleDirectResponse(userInput: string, history: string[]): string | null {
    if (history.length === 0) return null;
    
    const lowerInput = userInput.toLowerCase().trim().replace(/[.,!?]/g, '');
    const lastBotMessage = history[history.length - 1].toLowerCase();

    if (lastBotMessage.includes("я правильно тебя понял?")) {
        if (lowerInput === "нет" || lowerInput === "неправильно") {
            return "Понял, моя ошибка. Попробуй, пожалуйста, перефразировать свой вопрос, чтобы я лучше понял.";
        }
        if (lowerInput === "да" || lowerInput === "правильно") {
            return "Отлично! Рад, что мы на одной волне.";
        }
    }

    return null;
}

// --- Pipeline Stage 2: Learning Algorithm ---

/**
 * Checks if the user is defining the last unknown word.
 * @param userInput The user's message.
 * @returns The definition if it's a definition, otherwise null.
 */
function isDefiningUnknownWord(userInput: string): string | null {
  if (!sessionMemory.lastUnknownWord) {
    return null;
  }
  const lowerInput = userInput.toLowerCase();
  const patterns = [
    new RegExp(`^${sessionMemory.lastUnknownWord}\\s*-\\s*(.+)`),
    new RegExp(`^${sessionMemory.lastUnknownWord}\\s*это\\s*(.+)`),
    new RegExp(`^${sessionMemory.lastUnknownWord}\\s*значит\\s*(.+)`),
  ];

  for (const pattern of patterns) {
    const match = lowerInput.match(pattern);
    if (match && match[1]) {
      return match[1].trim();
    }
  }
  return null;
}

/**
 * Asynchronously saves a new learned word to the JSON file.
 * @param word The word that was learned.
 * @param definition The definition of the word.
 */
async function saveLearnedWord(word: string, definition: string): Promise<void> {
  const filePath = path.join(process.cwd(), 'src', 'data', 'learned-words.json');
  const updatedLearnedWords = {...lw, [word]: definition};

  try {
    await new Promise<void>((resolve, reject) => {
      fs.writeFile(
        filePath,
        JSON.stringify(updatedLearnedWords, null, 2),
        err => {
          if (err) {
            console.error('Failed to write to learned-words.json', err);
            reject(err);
          } else {
            Object.assign(lw, updatedLearnedWords);
            resolve();
          }
        }
      );
    });
  } catch (error) {
    // Error is already logged
  }
}

// --- Pipeline Stage 3: Conversational Context Analyzer ---

const personalPronounMarkers = ['я', 'у меня', 'мой', 'мне', 'я сам'];
const questionAboutWellbeing = [
  'как дела', 'как ты', 'как поживаешь', 'как твое', 'как сам', 'как настроение',
];
const positiveUserStates = [
  'хорошо', 'нормально', 'отлично', 'замечательно', 'прекрасно', 'в порядке', 'ничего', 'пойдет',
];

/**
 * Checks if the user is likely responding to a question about their well-being.
 * @param userInput The user's current message.
 * @param history The conversation history.
 * @returns true if it's likely a personal response to a well-being question.
 */
function isPersonalResponseToWellbeing(userInput: string, history: string[]): boolean {
  if (history.length === 0) return false;

  const lowerInput = userInput.toLowerCase().replace(/[.,!?]/g, '').trim();
  const lastBotMessage = history[history.length - 1].toLowerCase();

  const wasAsked = questionAboutWellbeing.some(q => lastBotMessage.includes(q));
  if (!wasAsked) {
    return false;
  }

  const isPersonal = personalPronounMarkers.some(marker =>
    lowerInput.startsWith(marker)
  );
  
  const isDirectAnswer = positiveUserStates.some(state => lowerInput.includes(state));

  return isPersonal || isDirectAnswer;
}


// --- Pipeline Stage 4: Knowledge Base Matcher ---

/**
 * Finds the best matching intent from the knowledge base using direct and fuzzy matching.
 * @param lowerCaseInput The user's input in lower case.
 * @returns The best match object or null.
 */
function findBestIntentMatch(lowerCaseInput: string): { intent: string; score: number } | null {
  let bestMatch: { intent: string; score: number } | null = null;
  for (const intent in kb) {
    if (intent === 'неизвестная_фраза' || !Object.prototype.hasOwnProperty.call(kb, intent)) continue;

    const intentData = kb[intent];
    for (const phrase of intentData.фразы) {
      const lowerPhrase = phrase.toLowerCase();
      
      if (lowerCaseInput.includes(lowerPhrase) && lowerPhrase.length > 0) {
        const score = lowerPhrase.length / lowerCaseInput.length;
        if (score > (bestMatch?.score ?? 0)) {
          bestMatch = { intent, score };
        }
      } else {
        const distance = levenshteinDistance(lowerCaseInput, lowerPhrase);
        const threshold = Math.floor(lowerPhrase.length * 0.4); // Stricter threshold
        if (distance <= threshold && lowerPhrase.length > 3) {
          const score = (1 - distance / lowerPhrase.length) * 0.9; // Fuzzy match has a penalty
          if (score > (bestMatch?.score ?? 0)) {
            bestMatch = { intent, score };
          }
        }
      }
    }
  }
  return bestMatch;
}


// --- Pipeline Stage 5: Creative Connection Algorithm ---

/**
 * Generates a response based on word connections if a direct intent is not found.
 * This simulates an "Attention Mechanism" by weighing words.
 * @param userInput The user's message.
 * @returns A generated response string or null if no connection is found.
 */
function generateConnectionResponse(userInput: string): string | null {
  const lowerCaseInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
  const words = [...new Set(lowerCaseInput.split(/\s+/).filter(w => w.length > 2))]; 
  const connections = wc.словарь_связей;
  const weights = wc.веса_частей_речи;

  const foundFacts: { word: string; fact: string; weight: number }[] = [];

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
        let hasFact = false;

        if (properties.is_a) {
          fact += ` - это ${Array.isArray(properties.is_a) ? properties.is_a.join(', ') : properties.is_a}`;
          hasFact = true;
        }
        if (properties.can_do) {
          fact += `${hasFact ? ', который' : ''} может ${Array.isArray(properties.can_do) ? properties.can_do.join(', ') : properties.can_do}`;
          hasFact = true;
        }
        if (properties.related_to) {
           fact += `. Связано с: ${Array.isArray(properties.related_to) ? properties.related_to.join(', ') : properties.related_to}`;
           hasFact = true;
        }
        
        if (hasFact) {
            fact += '.';
            foundFacts.push({ word, fact, weight: weights[partOfSpeech] || 1 });
        }
      }
    }
  }

  if (foundFacts.length === 0) {
    return null;
  }
  
  foundFacts.sort((a, b) => b.weight - a.weight);

  if (foundFacts.length === 1) {
    return foundFacts[0].fact;
  }

  let combinedResponse = 'Если я правильно тебя понимаю, ты говоришь о нескольких вещах. ';
  const factStrings = foundFacts.map(f => f.fact.replace(/.$/, ''));
  combinedResponse += factStrings.join(' и ');
  combinedResponse += '. Это довольно интересное сочетание.';

  return combinedResponse;
}


/**
 * The main generation function for Model Q.
 * It uses a pipeline of algorithms to generate the most relevant response.
 * @param userInput The user's message.
 * @param history The conversation history.
 * @returns A response string.
 */
async function generateCreativeResponse(
  userInput: string,
  history: string[] = []
): Promise<string> {
  const lowerCaseInput = userInput.toLowerCase().replace(/[.,!?]/g, '');
  const wordsInInput = lowerCaseInput.split(/\s+/).filter(w => w);

  // --- Start of Pipeline ---
  
  // 1. Direct Response check: Handle simple "yes/no" to bot's questions first.
  const directResponse = handleDirectResponse(userInput, history);
  if (directResponse) {
      return directResponse;
  }

  // 2. Learning Sub-algorithm: Check if user is defining a word
  const definition = isDefiningUnknownWord(userInput);
  if (definition && sessionMemory.lastUnknownWord) {
    const learnedWord = sessionMemory.lastUnknownWord;
    await saveLearnedWord(learnedWord, definition);
    sessionMemory.lastUnknownWord = null; // Clear state
    return `Понял! Теперь я знаю, что ${learnedWord} - это ${definition}. Спасибо!`;
  }
  sessionMemory.lastUnknownWord = null; // Reset if user moves on

  // 3. Contextual Sub-algorithm: Check for specific conversational contexts
  if (isPersonalResponseToWellbeing(userInput, history)) {
    const suitableResponses = [
      'Рад это слышать!', 'Отлично! Чем теперь займемся?', 'Здорово! Если что-то понадобится, я здесь.', 'Это хорошо. Что дальше по плану?',
    ];
    let responses = suitableResponses;
    const lastResponse = lastResponseMap.get('wellbeing_reply');
     if (lastResponse && responses.length > 1) {
      responses = responses.filter(r => r !== lastResponse);
    }
    const response = responses[Math.floor(Math.random() * responses.length)];
    lastResponseMap.set('wellbeing_reply', response);
    return synonymize(response);
  }
  
  // 4. Knowledge Base Sub-algorithm: Find a direct or fuzzy match
  const bestMatch = findBestIntentMatch(lowerCaseInput);

  if (bestMatch && bestMatch.score > 0.5) { // Stricter confidence threshold
    let responses = kb[bestMatch.intent].ответы;
    const lastResponse = lastResponseMap.get(bestMatch.intent);
    if (lastResponse && responses.length > 1) {
      responses = responses.filter(r => r !== lastResponse);
    }
    const response = responses[Math.floor(Math.random() * responses.length)];
    lastResponseMap.set(bestMatch.intent, response);
    
    // Clarify only if the match was fuzzy (not a direct full phrase match)
    if (bestMatch.score < 0.95 && bestMatch.score > 0.5) {
      return synonymize(response) + ' Я правильно тебя понял?';
    }
    return synonymize(response);
  }

  // 5. Creative Connection Sub-algorithm: Use word connections
  const connectionResponse = generateConnectionResponse(userInput);
  if (connectionResponse) {
    return synonymize(connectionResponse);
  }
  
  // 6. Learned Knowledge Sub-algorithm: Check persistent learned words
  const learnedMeaning = lw[lowerCaseInput as keyof typeof lw];
  if (learnedMeaning) {
    return `Я помню, ты говорил, что ${lowerCaseInput} - это ${learnedMeaning}. Хочешь поговорить об этом?`;
  }

  // 7. Fallback Sub-algorithm: Thoughtful default response with learning trigger
  const thoughtfulResponses = [
    `Я размышляю над словом "${
      wordsInInput.find(w => w.length > 3) || wordsInInput[0] || 'это'
    }"... и что оно может означать в этом контексте. Можешь объяснить?`,
    'Это сложный вопрос. Мои алгоритмы ищут наиболее подходящий ответ, но пока безуспешно. Попробуешь перефразировать?',
  ];
  let responses = thoughtfulResponses;
  const lastResponse = lastResponseMap.get('fallback');
  if (lastResponse && responses.length > 1) {
      responses = responses.filter(r => r !== lastResponse);
  }
  const response = responses[Math.floor(Math.random() * responses.length)];
  lastResponseMap.set('fallback', response);

  const mentionedWord = response.match(/"(.*?)"/);
  if (mentionedWord && mentionedWord[1]) {
     sessionMemory.lastUnknownWord = mentionedWord[1];
  }

  return synonymize(response);
}


// --- Main Entry Point ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  const history = input.history || [];
  const aiResponse = await generateCreativeResponse(input.userInput, history);
  return {aiResponse};
}
