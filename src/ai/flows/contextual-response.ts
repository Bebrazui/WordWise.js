
'use server';
/**
 * @fileOverview A contextual response AI bot based on a knowledge base, synonyms, and word connections.
 * This file implements the "Bot Q (Quantum)" model, which uses a pipeline of algorithms to generate responses.
 *
 * - contextualResponse - The main entry point function that handles the response generation process.
 * - ContextualResponseInput - The input type for the contextualResponse function.
 * - ContextualResponseOutput - The return type for the contextualResponse function.
 */

import {z} from 'zod';
import knowledgeBase from '@/data/knowledge-base.json';
import synonyms from '@/data/synonyms.json';
import wordConnections from '@/data/word-connections.json';
import learnedWords from '@/data/learned-words.json';
import morphology from '@/data/morphology.json';

// In a real-world scenario, this would be a proper database write.
// For this environment, we simulate it, but it won't persist across server restarts.
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
  experimental: z.boolean().optional().describe('Flag to enable experimental response generation.')
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
    phrases: string[];
    responses: string[];
  };
};

type Synonyms = {
  [key: string]: string[];
};

type WordConnection = {
    value: string | string[];
    weight: number;
    type: string;
}

type WordConnections = {
  [word: string]: WordConnection[];
};

type LearnedWords = {
  [word:string]: string;
};

type Morphology = {
    prefixes: string[];
    suffixes: Record<string, string[]>;
    endings: string[];
};


// --- Load Knowledge Bases ---
const kb = knowledgeBase as KnowledgeBase;
const syn = synonyms as Synonyms;
const wc = wordConnections as WordConnections;
const lw = learnedWords as LearnedWords;
const morph = morphology as Morphology;


// --- State Management ---
const lastResponseMap = new Map<string, string>();
const sessionMemory = {
  lastUnknownWord: null as string | null,
};

// =================================================================
// === ALGORITHM PIPELINE FOR "Bot Q (Quantum)" ====================
// =================================================================

// --- Pipeline Stage 1: Input Cleaner & Lemmatizer ---
/**
 * Simple rule-based lemmatizer/stemmer. It's not perfect but helps handle basic word forms.
 * It now prioritizes checking for a direct match in synonyms before attempting to stem.
 * @param word The word to lemmatize.
 * @returns The lemmatized (base) form of the word.
 */
function lemmatize(word: string): string {
    word = word.toLowerCase();
    // 1. Check if the word is already a known base form or a direct synonym key.
    if(syn[word] || wc[word] || kb[word] || lw[word as keyof typeof lw]) return word;
    
    // 2. Check if the word is a known synonym form of a base word.
    for(const key in syn){
        if(syn[key].includes(word)) return key;
    }

    // 3. Adjective endings (e.g., "хорошее" -> "хороший")
    const adjEndings = ['ый', 'ий', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие', 'его', 'ого', 'ему', 'ому', 'ую', 'юю', 'ой', 'ей'];
    for (const ending of adjEndings) {
        if (word.endsWith(ending)) {
            let base = word.slice(0, -ending.length);
            // Attempt to form a common masculine adjective
            if (syn[base + 'ий'] || Object.values(syn).flat().includes(base + 'ий')) return base + 'ий';
            if (syn[base + 'ый'] || Object.values(syn).flat().includes(base + 'ый')) return base + 'ый';
            // A common case: "хорошее" -> "хорош" -> "хорошо" (adverb)
            if (syn[base + 'о']) return base + 'о';
             // Default to a plausible base form if others fail
            if (syn[base + 'ий'] || wc[base + 'ий']) return base + 'ий';
            if (syn[base + 'ый'] || wc[base + 'ый']) return base + 'ый';
            return base + 'ий';
        }
    }

    // 4. Noun endings
    const nounEndings = ['а', 'у', 'е', 'ы', 'ом', 'ам', 'ах', 'ой', 'ей', 'ю', 'ями', 'ами'];
    for (const ending of nounEndings) {
        if (word.length > 3 && word.endsWith(ending)) {
            let base = word.slice(0, -ending.length);
            // If the base form exists in our knowledge, use it.
            if (kb[base] || wc[base] || syn[base]) return base;
            // Try adding a default ending if the direct base is not found
            if(syn[base] || wc[base]) return base;
        }
    }
    
    // 5. Verb endings
    const verbEndings = ['ешь', 'ет', 'ем', 'ете', 'ут', 'ют', 'ишь', 'ит', 'им', 'ите', 'ат', 'ят', 'л', 'ла', 'ло', 'ли', 'ть', 'ти'];
    for (const ending of verbEndings) {
        if(word.endsWith(ending)) {
            let base = word.slice(0, -ending.length);
            if(syn[base + 'ть'] || wc[base + 'ть'] || lw[base + 'ть' as keyof typeof lw]) return base + 'ть';
            if(syn[base + 'ти'] || wc[base + 'ти'] || lw[base + 'ти' as keyof typeof lw]) return base + 'ти';
        }
    }

    return word; // return original if no rule matched
}


/**
 * Prepares user input for processing by cleaning, normalizing, and lemmatizing it.
 * @param userInput The raw user input.
 * @returns A cleaned, lowercased string of lemmatized words.
 */
function cleanAndNormalizeInput(userInput: string): string {
  if (!userInput) return '';
  const cleaned = userInput.toLowerCase().trim().replace(/[.,!?]/g, '');
  const words = cleaned.split(/\s+/);
  const lemmatizedWords = words.map(word => lemmatize(word));
  return lemmatizedWords.join(' ');
}


/**
 * Calculates the Levenshtein distance between two strings.
 * Used for fuzzy matching of phrases.
 * @param a The first string.
 * @param b The second string.
 * @returns The Levenshtein distance.
 */
function levenshteinDistance(a: string, b: string): number {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;
    const matrix = Array.from(Array(a.length + 1), () => Array(b.length + 1).fill(0));
    for (let i = 0; i <= a.length; i++) { matrix[i][0] = i; }
    for (let j = 0; j <= b.length; j++) { matrix[0][j] = j; }
    for (let i = 1; i <= a.length; i++) {
        for (let j = 1; j <= b.length; j++) {
            const cost = a[i - 1] === b[j - 1] ? 0 : 1;
            matrix[i][j] = Math.min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            );
        }
    }
    return matrix[a.length][b.length];
}

// --- Pipeline Stage 2: Learning Algorithm ---
/**
 * Checks if the user is defining the last unknown word.
 * @param userInput The user's raw input.
 * @returns The definition if it's a definition, otherwise null.
 */
function checkForDefinition(userInput: string): string | null {
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
            Object.assign(lw, updatedLearnedWords); // Update in-memory cache
            resolve();
          }
        }
      );
    });
  } catch (error) {
    // Error is already logged
  }
}

// --- Pipeline Stage 3: Direct Response Handler ---
/**
 * Handles simple, direct responses to bot's questions, like "нет" to "Я правильно тебя понял?".
 * This is the first check to avoid misinterpretation loops.
 * @param normalizedInput The user's cleaned message.
 * @param history The conversation history.
 * @returns A specific response or null if no direct response context is found.
 */
function handleDirectResponse(normalizedInput: string, history: string[]): string | null {
    if (history.length === 0) return null;
    
    const lastBotMessage = history[history.length - 1].toLowerCase();

    if (lastBotMessage.includes("я правильно тебя понял?")) {
        if (normalizedInput === "нет" || normalizedInput === "неправильно") {
            return "Понял, моя ошибка. Попробуй, пожалуйста, перефразировать свой вопрос, чтобы я лучше понял.";
        }
        if (normalizedInput === "да" || normalizedInput === "правильно" || normalizedInput === "ага") {
            return "Отлично! Рад, что мы на одной волне.";
        }
    }
    return null;
}

// --- Pipeline Stage 4: Contextual Analyzer ---
const questionAboutWellbeing = [
  'как дела', 'как ты', 'как поживаешь', 'как твое', 'как сам', 'как настроение', 'что нового', 'как самочувствие', 'как твои дела'
];
const positiveUserStates = [
  'хорошо', 'нормально', 'отлично', 'замечательно', 'прекрасно', 'порядок', 'ничего', 'пойдет', 'хороший', 'у меня все хорошо', 'все хорошо', 'все отлично', 'все прекрасно'
];

/**
 * Checks if the user is likely responding to a question about their well-being.
 * @param normalizedInput The user's cleaned message (lemmatized).
 * @param history The conversation history.
 * @returns A relevant response or null.
 */
function handleWellbeingResponse(normalizedInput: string, history: string[]): string | null {
    if (history.length === 0) return null;

    const lastBotMessage = history[history.length - 1].toLowerCase();
    const wasAsked = questionAboutWellbeing.some(q => lastBotMessage.includes(lemmatize(q)));
    if (!wasAsked) return null;
    
    // Check if the user's response is one of the positive states.
    const isPositiveResponse = positiveUserStates.some(state => normalizedInput.includes(state));

    if (isPositiveResponse) {
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
        return response;
    }

    return null;
}

// --- Pipeline Stage 5: Knowledge Base Matcher (The Strategist) ---
/**
 * Finds the best matching intent from the knowledge base using direct and fuzzy matching.
 * @param normalizedInput The user's cleaned and lemmatized input.
 * @returns The best match object or null.
 */
function findBestIntentMatch(normalizedInput: string): { intent: string; score: number } | null {
  let bestMatch: { intent: string; score: number } | null = null;
  for (const intent in kb) {
    if (intent === 'unknown_phrase' || !Object.prototype.hasOwnProperty.call(kb, intent)) continue;

    const intentData = kb[intent];
    for (const phrase of intentData.phrases) {
      // KB phrases are assumed to be lemmatized
      const lowerPhrase = phrase.toLowerCase();
      
      // Direct phrase containment check
      if (normalizedInput.includes(lowerPhrase) && lowerPhrase.length > 0) {
        const score = lowerPhrase.length / normalizedInput.length;
        if (score > (bestMatch?.score ?? 0)) {
          bestMatch = { intent, score };
        }
      } 
      // Fuzzy match for similar phrases
      else {
        const distance = levenshteinDistance(normalizedInput, lowerPhrase);
        const threshold = Math.floor(lowerPhrase.length * 0.4); // Stricter threshold: 40% difference allowed
        if (distance <= threshold && lowerPhrase.length > 3) {
          const score = (1 - distance / lowerPhrase.length) * 0.9; // Fuzzy match has a 10% penalty
          if (score > (bestMatch?.score ?? 0)) {
            bestMatch = { intent, score };
          }
        }
      }
    }
  }
  return bestMatch;
}


// --- Pipeline Stage 6: Creative Connection Algorithm (The Quantum Generator) ---
/**
 * Generates a response based on word connections if a direct intent is not found.
 * This simulates an "Attention Mechanism" by weighing words and connection types.
 * @param normalizedInput The user's cleaned and lemmatized message.
 * @returns A generated response string or null if no connection is found.
 */
function generateConnectionResponse(normalizedInput: string): string | null {
    const words = [...new Set(normalizedInput.split(/\s+/).filter(w => w.length > 2))];
    if (words.length === 0) return null;

    const foundFacts: { fact: string; weight: number }[] = [];

    const allWordsAndSynonyms = new Set<string>();
    words.forEach(word => {
        allWordsAndSynonyms.add(word);
        const baseWord = syn[word] ? word : Object.keys(syn).find(key => syn[key].includes(word));
        if (baseWord && syn[baseWord]) {
            syn[baseWord].forEach(s => allWordsAndSynonyms.add(s));
        }
    });

    for (const word of allWordsAndSynonyms) {
        if (wc[word]) {
            const connections = [...wc[word]].sort((a, b) => b.weight - a.weight);
            let fact = `${word.charAt(0).toUpperCase() + word.slice(1)}`;
            let hasFact = false;
            let totalWeight = 0;
            
            for (const conn of connections.slice(0, 2)) { // Take top 2 connections for a word
                const connValue = Array.isArray(conn.value) ? conn.value.join(', ') : conn.value;
                let newPart = '';
                switch(conn.type) {
                    case 'is_a':
                        newPart = ` - это ${connValue}`;
                        break;
                    case 'can_do':
                        newPart = `${hasFact ? ', который' : ''} может ${connValue}`;
                        break;
                    case 'related_to':
                        newPart = `. Связано с: ${connValue}`;
                        break;
                    case 'property_of':
                        newPart = `. Является свойством: ${connValue}`;
                        break;
                    case 'antonym':
                        newPart = `, в противоположность - ${connValue}`;
                        break;
                }
                
                if (newPart) {
                    fact += newPart;
                    hasFact = true;
                    totalWeight += conn.weight;
                }
            }

            if (hasFact) {
                fact += '.';
                foundFacts.push({ fact, weight: totalWeight });
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

    // Combine multiple facts into a coherent sentence
    if (foundFacts.length > 1) {
        const fact1 = foundFacts[0].fact.replace(/\.$/, '');
        const fact2 = foundFacts[1].fact.toLowerCase();
        return `${fact1}, что, в свою очередь, связано с понятием "${fact2.split(' ')[0]}". Это интересная взаимосвязь.`;
    }

    return null;
}

// --- (Experimental) Pipeline Stage 6.5: Telegraphic Generator ---
/**
 * Generates a terse, "telegraphic" style response by linking keywords.
 * This is used for the experimental mode.
 * @param normalizedInput The user's cleaned and lemmatized message.
 * @returns A generated response string or null.
 */
function generateTelegraphicResponse(normalizedInput: string): string | null {
    const words = [...new Set(normalizedInput.split(/\s+/).filter(w => w.length > 2))];
    if (words.length === 0) return null;

    const keyConcepts: { word: string; connection: WordConnection | null }[] = [];

    // Find the most "important" words that have connections
    for (const word of words) {
        if (wc[word]) {
            const bestConnection = [...wc[word]].sort((a, b) => b.weight - a.weight)[0];
            keyConcepts.push({ word, connection: bestConnection });
        }
    }

    if (keyConcepts.length === 0) {
        return null;
    }

    // Sort concepts by connection weight to prioritize
    keyConcepts.sort((a, b) => (b.connection?.weight ?? 0) - (a.connection?.weight ?? 0));

    // Construct the telegraphic response
    let response = '';
    const mainConcept = keyConcepts[0];
    response += `${mainConcept.word.charAt(0).toUpperCase() + mainConcept.word.slice(1)}`;

    if (mainConcept.connection) {
        const conn = mainConcept.connection;
        const connValue = Array.isArray(conn.value) ? conn.value.join(', ') : conn.value;
        switch (conn.type) {
            case 'is_a':
                response += ` - это ${connValue}.`;
                break;
            case 'can_do':
                response += ` может ${connValue}.`;
                break;
            case 'related_to':
                 response += `. Связано с: ${connValue}.`;
                break;
            default:
                response += `: ${connValue}.`;
        }
    }
    
    // Add a second concept if available
    if (keyConcepts.length > 1) {
        const secondConcept = keyConcepts[1];
        if (secondConcept.connection) {
             response += ` Также: ${secondConcept.word} - ${secondConcept.connection.value}.`;
        }
    }

    return response.trim();
}


// --- Pipeline Stage 7: Observer & Smart Fallback ---
/**
 * Provides a thoughtful default response and triggers the learning mechanism.
 * @param normalizedInput The cleaned and lemmatized user input.
 * @returns A fallback response string.
 */
function getFallbackResponse(normalizedInput: string): string {
    const wordsInInput = normalizedInput.split(/\s+/).filter(w => w);
    // Try to find a word that is not a known synonym or in the knowledge base, and is longer than 2 characters
    let unknownWord = wordsInInput.find(w => 
        w.length > 2 && 
        !syn[w] && 
        !Object.keys(kb).some(k => kb[k].phrases.includes(w)) && 
        !lw[w as keyof typeof lw]
    );
    
    // If all words are known, pick the most complex one (longest) that is not a stop word
    if (!unknownWord && wordsInInput.length > 0) {
        unknownWord = wordsInInput
            .filter(w => w.length > 2)
            .sort((a, b) => b.length - a.length)[0];
    } 
    
    // If we still don't have a word (e.g., all words were short), we must have misunderstood.
    if (!unknownWord) {
        return kb.unknown_phrase.responses[Math.floor(Math.random() * kb.unknown_phrase.responses.length)];
    }


    const thoughtfulResponses = [
        `Я размышляю над словом "${unknownWord}"... и что оно может означать в этом контексте. Можешь объяснить? Например: ${unknownWord} - это...`,
        'Это сложный вопрос. Мои алгоритмы ищут наиболее подходящий ответ, но пока безуспешно. Попробуешь перефразировать?',
        `Хм, слово "${unknownWord}" поставило меня в тупик. Не мог бы ты его пояснить?`,
    ];
    let responses = thoughtfulResponses;
    const lastResponse = lastResponseMap.get('fallback');
    if (lastResponse && responses.length > 1) {
        responses = responses.filter(r => r !== lastResponse);
    }
    const response = responses[Math.floor(Math.random() * responses.length)];
    lastResponseMap.set('fallback', response);

    // Set the unknown word for the learning algorithm
    const mentionedWord = response.match(/"(.*?)"/);
    if (mentionedWord && mentionedWord[1]) {
        sessionMemory.lastUnknownWord = mentionedWord[1];
    } else {
        sessionMemory.lastUnknownWord = null;
    }

    return response;
}

/**
 * Dynamically replaces some words with their synonyms to make responses less repetitive.
 * @param sentence The sentence to process.
 * @returns A sentence with some words replaced by synonyms.
 */
function synonymize(sentence: string): string {
  const words = sentence.split(/(\s+|,|\.|\?|!)/);
  const newWords = words.map(word => {
    const lowerWord = word.toLowerCase();
    const synonymList = syn[lowerWord];
    // Replace with a 75% chance if synonyms exist
    if (synonymList && Math.random() < 0.75) { 
      const randomSynonym = synonymList[Math.floor(Math.random() * synonymList.length)];
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
 * The main generation function for Model Q.
 * It uses a pipeline of algorithms to generate the most relevant response.
 * @param userInput The user's message.
 * @param history The conversation history.
 * @param experimental Flag to force creative generation.
 * @returns A response string.
 */
async function generateCreativeResponse(
  userInput: string,
  history: string[] = [],
  experimental = false
): Promise<string> {
    
  // --- Start of Pipeline ---

  // Stage 1: Input Cleaner & Lemmatizer
  const normalizedInput = cleanAndNormalizeInput(userInput);
  if (!normalizedInput) {
      return "Пожалуйста, скажи что-нибудь.";
  }

  // Stage 2: Learning Algorithm
  const definition = checkForDefinition(userInput);
  if (definition && sessionMemory.lastUnknownWord) {
    const learnedWord = sessionMemory.lastUnknownWord;
    await saveLearnedWord(learnedWord, definition);
    sessionMemory.lastUnknownWord = null; // Clear state after learning
    return `Понял! Теперь я знаю, что ${learnedWord} - это ${definition}. Спасибо!`;
  }
  // Reset if user has moved on from defining a word
  if (sessionMemory.lastUnknownWord && !userInput.toLowerCase().startsWith(sessionMemory.lastUnknownWord)) {
      sessionMemory.lastUnknownWord = null;
  }
  
  // Stage 3: Direct Response Handler
  const directResponse = handleDirectResponse(normalizedInput, history);
  if (directResponse) {
      return directResponse;
  }

  // Stage 4: Contextual Analyzer (Well-being)
  const wellbeingResponse = handleWellbeingResponse(normalizedInput, history);
  if (wellbeingResponse) {
      return synonymize(wellbeingResponse);
  }
  
  // Experimental mode has a specific flow
  if (experimental) {
      const telegraphicResponse = generateTelegraphicResponse(normalizedInput);
      if (telegraphicResponse) {
          return telegraphicResponse;
      }
      // Fallback for experimental: try a normal KB match, but without synonymization
      const bestMatch = findBestIntentMatch(normalizedInput);
      if (bestMatch && bestMatch.score > 0.6) {
          let responses = kb[bestMatch.intent].responses;
          const response = responses[Math.floor(Math.random() * responses.length)];
          return response; // No synonymization
      }
      return getFallbackResponse(normalizedInput); // Final fallback if nothing works
  }

  // --- Normal flow continues here ---
  
  // Stage 5: Knowledge Base Matcher (The Strategist)
  const bestMatch = findBestIntentMatch(normalizedInput);

  if (bestMatch && bestMatch.score > 0.6) { // Confidence threshold
    let responses = kb[bestMatch.intent].responses;
    // Avoid repetition
    const lastResponse = lastResponseMap.get(bestMatch.intent);
    if (lastResponse && responses.length > 1) {
      responses = responses.filter(r => r !== lastResponse);
    }
    const response = responses[Math.floor(Math.random() * responses.length)];
    lastResponseMap.set(bestMatch.intent, response);
    
    // Clarify only if the match was fuzzy (not a direct, high-confidence match)
    if (bestMatch.score < 0.95 && bestMatch.score > 0.6) {
      return synonymize(response) + ' Я правильно тебя понял?';
    }
    return synonymize(response);
  }

  // Stage 6: Creative Connection Algorithm (The Quantum Generator)
  const connectionResponse = generateConnectionResponse(normalizedInput);
  if (connectionResponse) {
    return synonymize(connectionResponse);
  }
  
  // (Bonus Stage): Check persistent learned words
  const learnedMeaning = lw[normalizedInput as keyof typeof lw] || lw[lemmatize(normalizedInput) as keyof typeof lw];
  if (learnedMeaning) {
    const lemmatizedWord = lemmatize(normalizedInput);
    return `Я помню, ты говорил, что ${lemmatizedWord} - это ${learnedMeaning}. Хочешь поговорить об этом?`;
  }

  // Stage 7: Observer & Smart Fallback
  const fallbackResponse = getFallbackResponse(normalizedInput);
  return synonymize(fallbackResponse);
}


// --- Main Entry Point ---

export async function contextualResponse(
  input: ContextualResponseInput
): Promise<ContextualResponseOutput> {
  const history = input.history || [];
  const aiResponse = await generateCreativeResponse(input.userInput, history, input.experimental);
  return {aiResponse};
}
